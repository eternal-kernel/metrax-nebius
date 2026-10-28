"""Deterministic statutory validation module for Legal Metrology (Packaged Commodities) Rules, 2011."""

from typing import List, Optional
import re
from pydantic import BaseModel, Field


class ProductDeclarations(BaseModel):
    product_name: str
    net_quantity_raw: Optional[str] = None
    net_quantity_value: Optional[float] = None
    net_quantity_unit: Optional[str] = None
    mrp_raw: Optional[str] = None
    mrp_value: Optional[float] = None
    mrp_incl_taxes_stated: bool = False
    unit_sale_price_declared: Optional[float] = None
    unit_sale_price_basis: Optional[str] = None
    manufacturer_or_importer_address: Optional[str] = None
    country_of_origin: Optional[str] = None
    consumer_care_details: Optional[str] = None
    month_and_year_of_manufacture: Optional[str] = None


class RuleViolation(BaseModel):
    rule_citation: str
    field_name: str
    severity: str  # "CRITICAL", "MAJOR", or "WARNING"
    issue_description: str
    statutory_requirement: str


class DeterministicAuditResult(BaseModel):
    compliance_score: int  # 0 to 100
    status: str  # "COMPLIANT", "NON_COMPLIANT", or "NEEDS_ULTRA_ESCALATION"
    requires_deep_reasoning: bool
    violations: List[RuleViolation] = Field(default_factory=list)
    passed_checks: List[str] = Field(default_factory=list)
    computed_unit_price_check: Optional[str] = None


PERMITTED_METRIC_UNITS = {
    "g",
    "gm",
    "gram",
    "grams",
    "kg",
    "kilogram",
    "ml",
    "millilitre",
    "l",
    "litre",
    "liter",
    "cm",
    "m",
    "number",
    "nos",
    "pcs",
    "pieces",
    "unit",
    "units",
}

VAGUE_QUALIFIERS = ["approx", "about", "around", "estimated"]


def evaluate_declarations(decl: ProductDeclarations) -> DeterministicAuditResult:
    """Evaluate product declarations against Legal Metrology (Packaged Commodities) Rules, 2011."""
    violations: List[RuleViolation] = []
    passed_checks: List[str] = []
    computed_unit_price_check: Optional[str] = None

    # 1. Rule 6(1)(a) [Manufacturer/Importer Address]
    addr = (decl.manufacturer_or_importer_address or "").strip()
    if not addr or len(addr) < 10:
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(a)",
                field_name="manufacturer_or_importer_address",
                severity="CRITICAL",
                issue_description=(
                    "Manufacturer/Packer/Importer complete name and address is missing or inadequate "
                    f"(found '{addr or 'None'}', minimum 10 characters required)."
                ),
                statutory_requirement=(
                    "Every package shall bear the name and complete address of the manufacturer, "
                    "packer, or importer."
                ),
            )
        )
    else:
        passed_checks.append(
            f"Rule 6(1)(a): Complete manufacturer/importer address declared ('{addr}')."
        )

    # 2. Rule 6(1)(aa) [Country of Origin]
    origin = (decl.country_of_origin or "").strip()
    if not origin:
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(aa)",
                field_name="country_of_origin",
                severity="CRITICAL",
                issue_description="Country of origin or manufacture is missing.",
                statutory_requirement=(
                    "The name of the country of origin or manufacture or assembly shall be clearly "
                    "mentioned on every pre-packaged commodity."
                ),
            )
        )
    else:
        passed_checks.append(
            f"Rule 6(1)(aa): Country of origin declared ('{origin}')."
        )

    # 3. Rule 6(1)(c) & Rule 12 [Net Quantity & SI Units]
    net_qty_passed = True
    if decl.net_quantity_value is None or decl.net_quantity_value <= 0:
        net_qty_passed = False
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(c) & Rule 12",
                field_name="net_quantity_value",
                severity="CRITICAL",
                issue_description="Net quantity numeric value is missing or non-positive.",
                statutory_requirement=(
                    "The net quantity in terms of standard unit of weight, measure or number must be "
                    "declared unambiguously."
                ),
            )
        )

    if decl.net_quantity_raw:
        raw_lower = decl.net_quantity_raw.lower()
        if any(q in raw_lower for q in VAGUE_QUALIFIERS):
            net_qty_passed = False
            violations.append(
                RuleViolation(
                    rule_citation="Rule 6(1)(c) & Rule 12",
                    field_name="net_quantity_raw",
                    severity="MAJOR",
                    issue_description=(
                        f"Net quantity statement '{decl.net_quantity_raw}' contains vague qualifier words "
                        "(e.g., 'approx', 'about', 'around', 'estimated')."
                    ),
                    statutory_requirement=(
                        "Declaration of net quantity shall not be qualified by words tending to exaggerate "
                        "or minimize the quantity."
                    ),
                )
            )

    unit_clean = (decl.net_quantity_unit or "").strip().lower()
    if not unit_clean or unit_clean not in PERMITTED_METRIC_UNITS:
        net_qty_passed = False
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(c) & Rule 12",
                field_name="net_quantity_unit",
                severity="MAJOR",
                issue_description=(
                    f"Unit of measurement '{decl.net_quantity_unit}' is non-standard. Legal Metrology rules "
                    "mandate SI / metric units."
                ),
                statutory_requirement=(
                    "Net quantity shall be expressed in terms of metric units conforming to the SI system "
                    "(g, kg, ml, l, cm, m, numbers/units)."
                ),
            )
        )

    if net_qty_passed:
        passed_checks.append(
            f"Rule 6(1)(c) & Rule 12: Standard net quantity declared ({decl.net_quantity_value} {decl.net_quantity_unit})."
        )

    # 4. Rule 6(1)(d) [Month & Year of Manufacture/Packing/Import]
    mfg_date = (decl.month_and_year_of_manufacture or "").strip()
    if not mfg_date:
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(d)",
                field_name="month_and_year_of_manufacture",
                severity="MAJOR",
                issue_description="Month and year of manufacture, packing, or import is missing.",
                statutory_requirement=(
                    "Every package shall bear the month and year in which the commodity is manufactured "
                    "or packed or imported."
                ),
            )
        )
    else:
        passed_checks.append(
            f"Rule 6(1)(d): Month and year of manufacture/packing declared ('{mfg_date}')."
        )

    # 5. Rule 6(1)(e) [MRP & Tax Inclusivity]
    mrp_ok = True
    if decl.mrp_value is None or decl.mrp_value <= 0:
        mrp_ok = False
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(e)",
                field_name="mrp_value",
                severity="CRITICAL",
                issue_description="Maximum Retail Price (MRP) numeric value is missing or invalid.",
                statutory_requirement=(
                    "The retail sale price of the package shall be declared clearly in Indian Rupees."
                ),
            )
        )

    if not decl.mrp_incl_taxes_stated:
        mrp_ok = False
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(1)(e)",
                field_name="mrp_incl_taxes_stated",
                severity="WARNING",
                issue_description="MRP declaration lacks the mandatory 'Inclusive of all taxes' statement.",
                statutory_requirement=(
                    "The retail sale price must explicitly state that it is inclusive of all taxes."
                ),
            )
        )

    if mrp_ok:
        passed_checks.append(
            f"Rule 6(1)(e): MRP declared as ₹{decl.mrp_value:.2f} inclusive of all taxes."
        )

    # 6. Rule 6(2) [Consumer Care Details]
    cc_details = (decl.consumer_care_details or "").strip()
    phone_digits = re.findall(r"\d", cc_details)
    has_email = "@" in cc_details
    has_phone = len(phone_digits) >= 7

    if not cc_details or not (has_email or has_phone):
        violations.append(
            RuleViolation(
                rule_citation="Rule 6(2)",
                field_name="consumer_care_details",
                severity="MAJOR",
                issue_description=(
                    "Consumer care details missing, incomplete, or lack a valid phone number (>= 7 digits) "
                    "or email address."
                ),
                statutory_requirement=(
                    "Every package shall mention the name, address, telephone number, and email address of "
                    "the designated consumer complaint redressal officer."
                ),
            )
        )
    else:
        passed_checks.append(
            "Rule 6(2): Consumer care contact details verified with valid contact channel."
        )

    # 7. Rule 6(11) [Unit Sale Price Mathematical Verification]
    if (
        decl.mrp_value is not None
        and decl.mrp_value > 0
        and decl.net_quantity_value is not None
        and decl.net_quantity_value > 0
    ):
        basis = (decl.unit_sale_price_basis or "").strip().lower()
        is_per_100 = "100" in basis

        if is_per_100:
            expected_unit_price = (
                decl.mrp_value / decl.net_quantity_value
            ) * 100.0
            basis_str = (
                f"per 100 {decl.net_quantity_unit}"
                if decl.net_quantity_unit
                else "per 100g/ml"
            )
        else:
            expected_unit_price = decl.mrp_value / decl.net_quantity_value
            basis_str = f"per 1 {decl.net_quantity_unit or 'unit'}"

        expected_unit_price = round(expected_unit_price, 2)

        if decl.unit_sale_price_declared is None:
            violations.append(
                RuleViolation(
                    rule_citation="Rule 6(11)",
                    field_name="unit_sale_price_declared",
                    severity="MAJOR",
                    issue_description=(
                        f"Unit sale price declaration is missing. Statutorily calculated unit price is "
                        f"₹{expected_unit_price:.2f} ({basis_str})."
                    ),
                    statutory_requirement=(
                        "Declaration of Unit Sale Price is mandatory for pre-packaged commodities to enable "
                        "informed price comparisons."
                    ),
                )
            )
            computed_unit_price_check = (
                f"Calculated expected unit price: ₹{expected_unit_price:.2f} ({basis_str}). "
                "Package lacks unit sale price declaration."
            )
        else:
            declared_price = round(decl.unit_sale_price_declared, 2)
            relative_diff = (
                abs(declared_price - expected_unit_price) / expected_unit_price
                if expected_unit_price > 0
                else 0.0
            )

            if relative_diff > 0.05:
                violations.append(
                    RuleViolation(
                        rule_citation="Rule 6(11)",
                        field_name="unit_sale_price_declared",
                        severity="CRITICAL",
                        issue_description=(
                            "Mathematical Unit Price Discrepancy / Shrinkflation Misrepresentation: "
                            f"Declared unit price is ₹{declared_price:.2f} ({basis_str}), but true calculated "
                            f"price is ₹{expected_unit_price:.2f} ({basis_str}) derived from MRP ₹{decl.mrp_value:.2f} "
                            f"and net quantity {decl.net_quantity_value} {decl.net_quantity_unit or ''} "
                            f"(discrepancy of {relative_diff * 100:.1f}%)."
                        ),
                        statutory_requirement=(
                            "Unit sale price declared on the package must accurately reflect the mathematical "
                            "quotient of MRP divided by net quantity."
                        ),
                    )
                )
                computed_unit_price_check = (
                    f"Discrepancy detected: Declared ₹{declared_price:.2f} vs Computed ₹{expected_unit_price:.2f} "
                    f"({basis_str}), discrepancy of {relative_diff * 100:.1f}%."
                )
            else:
                passed_checks.append(
                    f"Rule 6(11): Declared unit sale price ₹{declared_price:.2f} matches calculated price "
                    f"₹{expected_unit_price:.2f} ({basis_str}) within 5% tolerance."
                )
                computed_unit_price_check = (
                    f"Mathematical audit passed: Declared unit price ₹{declared_price:.2f} matches "
                    f"computed price ₹{expected_unit_price:.2f} ({basis_str})."
                )
    else:
        computed_unit_price_check = (
            "Unable to compute unit sale price due to missing or invalid MRP or Net Quantity."
        )

    # Compute compliance score: Start at 100, deduct 20 for CRITICAL, 12 for MAJOR, 5 for WARNING
    deductions = {
        "CRITICAL": 20,
        "MAJOR": 12,
        "WARNING": 5,
    }
    score = 100
    for v in violations:
        score -= deductions.get(v.severity.upper(), 10)
    score = max(0, score)

    # Determine status & deep reasoning flag
    has_critical = any(v.severity == "CRITICAL" for v in violations)
    if len(violations) == 0:
        status = "COMPLIANT"
        requires_deep_reasoning = False
    elif has_critical or score < 70:
        status = "NON_COMPLIANT"
        requires_deep_reasoning = True
    else:
        status = "NEEDS_ULTRA_ESCALATION"
        requires_deep_reasoning = True

    return DeterministicAuditResult(
        compliance_score=score,
        status=status,
        requires_deep_reasoning=requires_deep_reasoning,
        violations=violations,
        passed_checks=passed_checks,
        computed_unit_price_check=computed_unit_price_check,
    )
