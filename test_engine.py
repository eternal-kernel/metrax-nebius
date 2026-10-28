"""End-to-End multi-model routing and inspection test suite for Phase 3."""

import sys
from engine import (
    MODEL_NANO,
    MODEL_ULTRA,
    run_full_inspection,
)

# ---------------------------------------------------------------------
# Sample 1: 100% Compliant Raw E-Commerce Food Listing Text
# ---------------------------------------------------------------------
SAMPLE_1_COMPLIANT_TEXT = """
Title: Fortune Sunlite Refined Sunflower Oil, 1 L Pouch
Brand: Fortune
Manufacturer & Packer: Adani Wilmar Limited, Fortune House, Near Navrangpura Railway Crossing, Ahmedabad - 380009, Gujarat, India.
Net Quantity: 1 L
MRP: Rs. 145.00 (Inclusive of all taxes)
Unit Sale Price: Rs. 145.00 per L
Country of Origin: India
Month and Year of Packaging: 10/2026
Customer Complaint Contact: Executive - Consumer Care, Adani Wilmar Ltd., Email: customercare@adaniwilmar.in, Toll Free Phone: 1800-233-9999
Storage Instructions: Store in a cool and dry place away from direct sunlight.
"""

# ---------------------------------------------------------------------
# Sample 2: Non-Compliant Unstructured Supplement Listing Text
# ---------------------------------------------------------------------
SAMPLE_2_NON_COMPLIANT_TEXT = """
Title: Titan Beast High Protein Mass Gainer - Dutch Chocolate Flavor
Description: Premium muscle fuel imported directly from our global manufacturing partner.
Net Content: Approx 32 oz
Price: MRP Rs. 4,800 only! Special festival deal!
Unit Price: Rs. 100 per oz
Packed by: Titan Nutrition LLC, 500 Fifth Ave, NY, USA
Imported by: Titan India Impex Pvt Ltd
Customer Support: Message our Instagram @titan_beast or call our desk
Batch No: TB-9921, Pkd Date: 08/2026
Usage: Mix 2 scoops with milk daily.
"""


def print_telemetry_table(telemetry_steps):
    print("\n  +" + "-" * 32 + "+" + "-" * 38 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 14 + "+")
    print(f"  | {'Pipeline Step':<30} | {'Model / Tool Engine':<36} | {'Triggered':<9} | {'Latency':<10} | {'Est. Cost':<12} |")
    print("  +" + "-" * 32 + "+" + "-" * 38 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 14 + "+")
    for s in telemetry_steps:
        trig_str = "YES" if s.triggered else "BYPASS"
        lat_str = f"{s.latency_ms} ms" if s.triggered else "0 ms"
        cost_str = f"${s.estimated_cost_usd:.5f}" if s.triggered else "$0.00000"
        tool_display = s.model_or_tool.split("/")[-1] if "/" in s.model_or_tool else s.model_or_tool
        print(f"  | {s.step_name:<30} | {tool_display:<36} | {trig_str:<9} | {lat_str:<10} | {cost_str:<12} |")
    print("  +" + "-" * 32 + "+" + "-" * 38 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 14 + "+")


def test_sample_1_compliant():
    print("\n" + "=" * 76)
    print("TEST SAMPLE 1: COMPLIANT RAW E-COMMERCE LISTING")
    print("=" * 76)

    report = run_full_inspection(SAMPLE_1_COMPLIANT_TEXT, is_url=False, force_ultra=False)

    print(f"\n1. Extracted Declarations (via Nemotron-3-Nano):")
    print(f"   - Product Name           : {report.extracted_declarations.product_name}")
    print(f"   - Net Quantity           : {report.extracted_declarations.net_quantity_value} {report.extracted_declarations.net_quantity_unit} (Raw: '{report.extracted_declarations.net_quantity_raw}')")
    print(f"   - MRP                    : ₹{report.extracted_declarations.mrp_value} (Tax Inclusive Stated: {report.extracted_declarations.mrp_incl_taxes_stated})")
    print(f"   - Unit Sale Price        : ₹{report.extracted_declarations.unit_sale_price_declared} ({report.extracted_declarations.unit_sale_price_basis})")
    print(f"   - Country of Origin      : {report.extracted_declarations.country_of_origin}")
    print(f"   - Manufacturer / Address : {report.extracted_declarations.manufacturer_or_importer_address}")
    print(f"   - Consumer Care          : {report.extracted_declarations.consumer_care_details}")

    print(f"\n2. Deterministic Statutory Audit:")
    print(f"   - Compliance Score       : {report.deterministic_audit.compliance_score}/100")
    print(f"   - Status                 : {report.deterministic_audit.status}")
    print(f"   - Requires Ultra Tier    : {report.deterministic_audit.requires_deep_reasoning}")
    print(f"   - Unit Price Math Check  : {report.deterministic_audit.computed_unit_price_check}")
    print(f"   - Violations Count       : {len(report.deterministic_audit.violations)}")
    print(f"   - Passed Checks Count    : {len(report.deterministic_audit.passed_checks)}")

    print(f"\n3. Smart Router Routing Decision & Telemetry:")
    print_telemetry_table(report.telemetry_steps)
    print(f"\n   Total Pipeline Latency   : {report.total_latency_ms} ms")
    print(f"   Estimated Compute Savings: {report.estimated_cost_savings_pct}% (Nemotron Ultra 550B bypassed!)")

    # Assertions
    assert report.deterministic_audit.compliance_score == 100, f"Expected 100, got {report.deterministic_audit.compliance_score}"
    assert report.deterministic_audit.status == "COMPLIANT", f"Expected COMPLIANT, got {report.deterministic_audit.status}"
    ultra_step = next((s for s in report.telemetry_steps if s.model_or_tool == MODEL_ULTRA), None)
    assert ultra_step is not None and not ultra_step.triggered, "Nemotron Ultra should have been bypassed for compliant product"
    assert report.estimated_cost_savings_pct > 80.0, "Cost savings should exceed 80% when Ultra is bypassed"
    print("\n[SUCCESS] Sample 1 PASSED: Fast Nano extraction, 100% compliance, and Ultra successfully bypassed!")


def test_sample_2_non_compliant():
    print("\n" + "=" * 76)
    print("TEST SAMPLE 2: NON-COMPLIANT UNSTRUCTURED SUPPLEMENT LISTING")
    print("=" * 76)

    report = run_full_inspection(SAMPLE_2_NON_COMPLIANT_TEXT, is_url=False, force_ultra=False)

    print(f"\n1. Extracted Declarations (via Nemotron-3-Nano):")
    print(f"   - Product Name           : {report.extracted_declarations.product_name}")
    print(f"   - Net Quantity           : {report.extracted_declarations.net_quantity_value} {report.extracted_declarations.net_quantity_unit} (Raw: '{report.extracted_declarations.net_quantity_raw}')")
    print(f"   - MRP                    : ₹{report.extracted_declarations.mrp_value} (Tax Inclusive Stated: {report.extracted_declarations.mrp_incl_taxes_stated})")
    print(f"   - Unit Sale Price        : ₹{report.extracted_declarations.unit_sale_price_declared} ({report.extracted_declarations.unit_sale_price_basis})")
    print(f"   - Country of Origin      : {report.extracted_declarations.country_of_origin}")
    print(f"   - Consumer Care          : {report.extracted_declarations.consumer_care_details}")

    print(f"\n2. Deterministic Statutory Audit (Violations Caught):")
    print(f"   - Compliance Score       : {report.deterministic_audit.compliance_score}/100")
    print(f"   - Status                 : {report.deterministic_audit.status}")
    print(f"   - Requires Ultra Tier    : {report.deterministic_audit.requires_deep_reasoning}")
    print(f"   - Math Discrepancy Check : {report.deterministic_audit.computed_unit_price_check}")
    print(f"   - Total Violations       : {len(report.deterministic_audit.violations)}")
    for v in report.deterministic_audit.violations:
        print(f"     * [{v.severity}] {v.rule_citation} ({v.field_name}): {v.issue_description}")

    print(f"\n3. Live Tavily Regulatory Citations Fetched ({len(report.tavily_citations)}):")
    for c in report.tavily_citations[:2]:
        print(f"   - [{c.title}] ({c.url})")
        print(f"     Snippet: {c.snippet[:150]}...")

    print(f"\n4. Smart Router Routing Decision & Telemetry:")
    print_telemetry_table(report.telemetry_steps)
    print(f"\n   Total Pipeline Latency   : {report.total_latency_ms} ms")
    print(f"   Estimated Compute Savings: {report.estimated_cost_savings_pct}% (Nano used for extraction)")

    print(f"\n5. Nemotron-3-Ultra-550b Official Statutory Show-Cause Notice (Excerpt):")
    print("-" * 76)
    memo_preview = (report.ultra_legal_memo or "")[:1200]
    print(memo_preview + "\n[... Notice continues with Remediation Plan and Corrected Label Text ...]")
    print("-" * 76)

    # Assertions
    assert report.deterministic_audit.compliance_score < 70, f"Expected < 70, got {report.deterministic_audit.compliance_score}"
    assert report.deterministic_audit.status == "NON_COMPLIANT", f"Expected NON_COMPLIANT, got {report.deterministic_audit.status}"
    ultra_step = next((s for s in report.telemetry_steps if "Ultra" in s.model_or_tool or "Super" in s.model_or_tool and "Synthesis" in s.step_name), None)
    assert ultra_step is not None and ultra_step.triggered, "Deep legal synthesis should have been triggered"
    assert len(report.tavily_citations) > 0, "Tavily search citations should be present"
    assert report.ultra_legal_memo is not None and len(report.ultra_legal_memo) > 200, "Legal notice should be generated"
    print("\n[SUCCESS] Sample 2 PASSED: Nano extraction, deterministic violation catch, Tavily search, and Ultra legal synthesis completed!")


if __name__ == "__main__":
    test_sample_1_compliant()
    test_sample_2_non_compliant()
