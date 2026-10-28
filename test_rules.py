"""Test suite for Legal Metrology deterministic rules validation and Nebius model discovery."""

import os
from dotenv import load_dotenv
from openai import OpenAI

from rules import (
    ProductDeclarations,
    evaluate_declarations,
)

# Load environment variables
load_dotenv()


def run_tests():
    print("=" * 70)
    print("PHASE 2: STATUTORY COMPLIANCE DETERMINISTIC RULES TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test Case A: Fully Compliant Domestic Food Product
    # -------------------------------------------------------------
    print("\n>>> Running Test Case A: Compliant Product")
    product_a = ProductDeclarations(
        product_name="Organic Whole Wheat Atta",
        net_quantity_raw="5 kg",
        net_quantity_value=5.0,
        net_quantity_unit="kg",
        mrp_raw="Rs. 275.00 (Inclusive of all taxes)",
        mrp_value=275.0,
        mrp_incl_taxes_stated=True,
        unit_sale_price_declared=55.0,
        unit_sale_price_basis="per kg",
        manufacturer_or_importer_address="Himalayan Organics Ltd, Plot 42, Sector 18, Gurugram, Haryana - 122001",
        country_of_origin="India",
        consumer_care_details="Customer Support: 1800-123-4567, email: care@himalayanorganics.in",
        month_and_year_of_manufacture="10/2026",
    )

    result_a = evaluate_declarations(product_a)
    print(f"Product: {product_a.product_name}")
    print(f"Compliance Score: {result_a.compliance_score}/100")
    print(f"Status: {result_a.status}")
    print(f"Requires Deep Reasoning (Ultra Escalation): {result_a.requires_deep_reasoning}")
    print(f"Computed Unit Price Check: {result_a.computed_unit_price_check}")
    print(f"Violations Detected: {len(result_a.violations)}")
    print(f"Passed Checks ({len(result_a.passed_checks)}):")
    for chk in result_a.passed_checks:
        print(f"  [PASS] {chk}")

    # Assertions
    assert result_a.compliance_score == 100, f"Expected 100, got {result_a.compliance_score}"
    assert result_a.status == "COMPLIANT", f"Expected COMPLIANT, got {result_a.status}"
    assert result_a.requires_deep_reasoning is False, "Expected False for compliant product"
    assert len(result_a.violations) == 0, "Expected 0 violations"
    print("\n[SUCCESS] Test Case A PASSED flawlessly (Score: 100, Compliant).\n")

    # -------------------------------------------------------------
    # Test Case B: Violating Imported Dietary Supplement
    # -------------------------------------------------------------
    print("-" * 70)
    print(">>> Running Test Case B: Violating Imported Supplement")
    product_b = ProductDeclarations(
        product_name="Ultra Pure Whey Protein Powder",
        net_quantity_raw="Approx 16 oz",
        net_quantity_value=16.0,
        net_quantity_unit="oz",  # Non-SI metric unit violation
        mrp_raw="Rs. 3200",
        mrp_value=3200.0,
        mrp_incl_taxes_stated=False,  # Missing tax-inclusive statement
        unit_sale_price_declared=120.0,  # False math: 3200/16 = 200, declared is 120
        unit_sale_price_basis="per oz",
        manufacturer_or_importer_address="Global Health Inc, New York, NY, USA",
        country_of_origin=None,  # Missing country of origin
        consumer_care_details="Call support during business hours",  # Missing phone (>=7 digits) or email
        month_and_year_of_manufacture="05/2026",
    )

    result_b = evaluate_declarations(product_b)
    print(f"Product: {product_b.product_name}")
    print(f"Compliance Score: {result_b.compliance_score}/100")
    print(f"Status: {result_b.status}")
    print(f"Requires Deep Reasoning (Ultra Escalation): {result_b.requires_deep_reasoning}")
    print(f"Computed Unit Price Check: {result_b.computed_unit_price_check}")
    print(f"\nViolations Detected ({len(result_b.violations)}):")
    for idx, v in enumerate(result_b.violations, 1):
        print(f"\n  Violation #{idx}:")
        print(f"    - Citation: {v.rule_citation}")
        print(f"    - Field: {v.field_name}")
        print(f"    - Severity: [{v.severity}]")
        print(f"    - Issue: {v.issue_description}")
        print(f"    - Statutory Requirement: {v.statutory_requirement}")

    # Assertions
    assert result_b.compliance_score < 100, f"Expected < 100, got {result_b.compliance_score}"
    assert result_b.status == "NON_COMPLIANT", f"Expected NON_COMPLIANT, got {result_b.status}"
    assert result_b.requires_deep_reasoning is True, "Expected True for violating product"
    assert len(result_b.violations) >= 4, f"Expected at least 4 violations, got {len(result_b.violations)}"
    print("\n[SUCCESS] Test Case B PASSED (Accurately flagged all statutory violations).\n")

    # -------------------------------------------------------------
    # Nebius Token Factory: Query Available NVIDIA Models
    # -------------------------------------------------------------
    print("=" * 70)
    print("NEBIUS TOKEN FACTORY: DISCOVERING AVAILABLE NVIDIA MODELS")
    print("=" * 70)

    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    if not nebius_api_key:
        print("[WARNING] NEBIUS_API_KEY not found in environment.")
        return

    client = OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=nebius_api_key,
    )
    models = client.models.list()
    nvidia_models = sorted([m.id for m in models.data if "nvidia" in m.id.lower()])

    print(f"Found {len(nvidia_models)} NVIDIA models on Nebius Token Factory:\n")
    for m in nvidia_models:
        print(f"  * {m}")

    print("\nKey Model Identifiers identified for Phase 3:")
    nano_model = next((m for m in nvidia_models if "nano" in m.lower()), None)
    super_model = next((m for m in nvidia_models if "super" in m.lower() and "vl" not in m.lower()), None)
    ultra_model = next((m for m in nvidia_models if "ultra" in m.lower()), None)

    print(f"  - Fast Tier (Nano)  : {nano_model}")
    print(f"  - Logic Tier (Super): {super_model}")
    print(f"  - Deep Tier (Ultra) : {ultra_model}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
