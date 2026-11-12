"""Freddie Mac Single-Family Loan-Level Dataset schemas.

Names/order follow the Freddie Mac user guide. Layouts have changed across years,
so the loader reads the actual column count from each file and maps the first N
names. Columns beyond the known list become extra_<n>. VERIFY these against the
user guide for every vintage you load (the loader logs the layout it used).
"""

ORIGINATION_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy_status", "cltv", "dti", "orig_upb", "ltv",
    "orig_interest_rate", "channel", "ppm_flag", "amortization_type", "property_state",
    "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
    "orig_loan_term", "num_borrowers", "seller_name", "servicer_name",
    "super_conforming_flag", "pre_harp_loan_sequence_number", "program_indicator",
    "harp_indicator", "property_valuation_method", "io_indicator",
    "mi_cancellation_indicator",
]

PERFORMANCE_COLS = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
    "current_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity",
    "defect_settlement_date", "modification_flag", "zero_balance_code",
    "zero_balance_effective_date", "current_interest_rate", "current_deferred_upb",
    "ddlpi", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "expenses",
    "legal_costs", "maintenance_preservation_costs", "taxes_insurance_expenses",
    "miscellaneous_expenses", "actual_loss_calculation", "modification_cost",
    "step_modification_flag", "deferred_payment_modification", "estimated_ltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_status_code",
    "current_month_modification_cost", "interest_bearing_upb",
]

# Freddie codes meaning "not available" -> NULL (never imputed here).
ORIG_NULL_CODES = {
    "credit_score": "9999", "mi_pct": "999", "num_units": "99", "cltv": "999",
    "dti": "999", "ltv": "999", "num_borrowers": "99",
    "first_time_homebuyer_flag": "9", "occupancy_status": "9", "channel": "9",
    "property_type": "99", "loan_purpose": "9", "property_valuation_method": "9",
}
PERF_NULL_CODES = {"estimated_ltv": "999"}

ORIG_NUMERIC = {
    "credit_score": "INTEGER", "mi_pct": "INTEGER", "num_units": "INTEGER",
    "cltv": "INTEGER", "dti": "INTEGER", "ltv": "INTEGER", "orig_upb": "DOUBLE",
    "orig_interest_rate": "DOUBLE", "orig_loan_term": "INTEGER",
    "num_borrowers": "INTEGER",
}
PERF_NUMERIC = {
    "current_actual_upb": "DOUBLE", "loan_age": "INTEGER",
    "remaining_months_to_legal_maturity": "INTEGER",
    "current_interest_rate": "DOUBLE", "estimated_ltv": "INTEGER",
}
