
from typing import Annotated, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from churn.config import TARGET_ALIASES, TARGET_COLUMN
from churn.logger import get_logger

logger = get_logger(__name__)

YesNo = Literal["Yes", "No"]
NonNegFloat = Annotated[float, Field(ge=0)]
NonNegInt = Annotated[int, Field(ge=0)]
# Codigo de area: 3 letras da regiao + 3 letras da cidade + 3 digitos (ex.: SEAPOR503).
ServiceAreaCode = Annotated[str, Field(pattern=r"^[A-Z]{6}\d{3}$")]
# Preco do aparelho: numero inteiro ou "Unknown" (convertido em NaN + flag no features.py).
HandsetPriceValue = Annotated[str, Field(pattern=r"^(\d+|Unknown)$")]

MAX_REPORTED_ERRORS = 20


class CustomerFeatures(BaseModel):
    """Variaveis de um cliente usadas pelo modelo (entrada da API e base do CSV bruto).

    Campos que podem faltar na base original aceitam None e, por padrao, ficam ausentes:
    o pipeline os imputa pela mediana do treino.
    """

    model_config = ConfigDict(extra="ignore")

    # Uso e faturamento (podem vir ausentes no dataset bruto).
    MonthlyRevenue: float | None = None
    MonthlyMinutes: NonNegFloat | None = None
    TotalRecurringCharge: float | None = None
    DirectorAssistedCalls: NonNegFloat | None = None
    OverageMinutes: NonNegFloat | None = None
    RoamingCalls: NonNegFloat | None = None
    PercChangeMinutes: float | None = None
    PercChangeRevenues: float | None = None

    # Chamadas.
    DroppedCalls: NonNegFloat
    BlockedCalls: NonNegFloat
    UnansweredCalls: NonNegFloat
    CustomerCareCalls: NonNegFloat
    ThreewayCalls: NonNegFloat
    ReceivedCalls: NonNegFloat
    OutboundCalls: NonNegFloat
    InboundCalls: NonNegFloat
    PeakCallsInOut: NonNegFloat
    OffPeakCallsInOut: NonNegFloat
    DroppedBlockedCalls: NonNegFloat
    CallForwardingCalls: NonNegFloat
    CallWaitingCalls: NonNegFloat

    # Conta e aparelhos.
    MonthsInService: NonNegInt
    UniqueSubs: NonNegInt
    ActiveSubs: NonNegInt
    ServiceArea: ServiceAreaCode | None = None
    Handsets: NonNegFloat | None = None
    HandsetModels: NonNegFloat | None = None
    CurrentEquipmentDays: float | None = None
    AgeHH1: NonNegFloat | None = None
    AgeHH2: NonNegFloat | None = None

    # Perfil do cliente.
    ChildrenInHH: YesNo
    HandsetRefurbished: YesNo
    HandsetWebCapable: YesNo
    TruckOwner: YesNo
    RVOwner: YesNo
    Homeownership: Literal["Known", "Unknown"]
    BuysViaMailOrder: YesNo
    RespondsToMailOffers: YesNo
    OptOutMailings: YesNo
    NonUSTravel: YesNo
    OwnsComputer: YesNo
    HasCreditCard: YesNo
    NewCellphoneUser: YesNo
    NotNewCellphoneUser: YesNo
    ReferralsMadeBySubscriber: NonNegInt
    IncomeGroup: Annotated[int, Field(ge=0, le=9)]
    OwnsMotorcycle: YesNo
    AdjustmentsToCreditRating: NonNegInt
    HandsetPrice: HandsetPriceValue
    CreditRating: Literal[
        "1-Highest", "2-High", "3-Good", "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"
    ]
    PrizmCode: Literal["Other", "Rural", "Suburban", "Town"]
    Occupation: Literal[
        "Clerical", "Crafts", "Homemaker", "Other", "Professional", "Retired", "Self", "Student"
    ]
    MaritalStatus: Literal["Yes", "No", "Unknown"]

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Regras entre colunas que a base respeita e o pipeline assume."""

        if self.ActiveSubs > self.UniqueSubs:
            raise ValueError("ActiveSubs nao pode ser maior que UniqueSubs")
        if (
            self.Handsets is not None
            and self.HandsetModels is not None
            and self.HandsetModels > self.Handsets
        ):
            raise ValueError("HandsetModels nao pode ser maior que Handsets")
        if self.NewCellphoneUser == "Yes" and self.NotNewCellphoneUser == "Yes":
            raise ValueError("NewCellphoneUser e NotNewCellphoneUser nao podem ser ambos 'Yes'")
        return self


class ChurnRecord(CustomerFeatures):
    """Uma linha do dataset bruto: features + identificador, alvo e variaveis de retencao."""

    CustomerID: int
    Churn: YesNo
    # Variaveis de retencao: validadas no CSV, mas removidas do modelo (target leakage).
    RetentionCalls: NonNegInt
    RetentionOffersAccepted: NonNegInt
    MadeCallToRetentionTeam: YesNo


CHURN_RECORDS_ADAPTER = TypeAdapter(list[ChurnRecord])


def _format_errors(error: ValidationError, index: pd.Index) -> str:
    """Resume os erros de validação indicando a linha original do DataFrame."""

    errors = error.errors()
    lines = []
    for err in errors[:MAX_REPORTED_ERRORS]:
        position, *field = err["loc"]
        if field:
            column = ".".join(map(str, field))
            lines.append(f"linha {index[position]}, coluna {column!r}: {err['msg']} (valor={err.get('input')!r})")
        else:
            # Erros do model_validator valem para o registro inteiro: mostra so o CustomerID.
            customer_id = err["input"].get("CustomerID")
            lines.append(f"linha {index[position]}, CustomerID {customer_id}: {err['msg']}")
    if len(errors) > MAX_REPORTED_ERRORS:
        lines.append(f"... e mais {len(errors) - MAX_REPORTED_ERRORS} erros")
    return f"{len(errors)} erros de validação no dataset bruto:\n" + "\n".join(lines)


def validate_raw_churn_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o dataset bruto aceitando aliases conhecidos da coluna alvo."""

    target_column = next((column for column in TARGET_ALIASES if column in df.columns), None)
    if target_column is None:
        logger.error(f"Coluna alvo ausente. Use uma destas colunas: {TARGET_ALIASES}")
        raise ValueError(f"Coluna alvo ausente. Use uma destas colunas: {TARGET_ALIASES}")
    normalized = df.rename(columns={target_column: TARGET_COLUMN})

    missing = [name for name in ChurnRecord.model_fields if name not in normalized.columns]
    if missing:
        logger.error(f"Colunas ausentes no dataset bruto: {missing}")
        raise ValueError(f"Colunas ausentes no dataset bruto: {missing}")

    # Unicidade e regra da tabela inteira, por isso fica fora do ChurnRecord.
    duplicated_ids = normalized.loc[normalized["CustomerID"].duplicated(), "CustomerID"]
    if not duplicated_ids.empty:
        message = (
            f"{len(duplicated_ids)} CustomerID duplicados no dataset bruto: "
            f"{duplicated_ids.head(MAX_REPORTED_ERRORS).tolist()}"
        )
        logger.error(message)
        raise ValueError(message)

    # Pydantic não entende NaN como ausente; converte para None antes de validar.
    records = normalized.astype(object).where(normalized.notna(), None).to_dict(orient="records")
    try:
        validated = CHURN_RECORDS_ADAPTER.validate_python(records)
    except ValidationError as error:
        message = _format_errors(error, normalized.index)
        logger.error(message)
        raise ValueError(message) from error

    logger.info(f"Dataset bruto validado com sucesso: {len(validated)} linhas.")
    return pd.DataFrame(
        [record.model_dump() for record in validated],
        index=normalized.index,
        columns=list(ChurnRecord.model_fields),
    )
