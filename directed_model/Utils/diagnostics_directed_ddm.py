# =====================================================================================
# Checking R-hat and ESS values
def check_convergence(summary_df, rhat_thresh=1.01, ess_thresh=400):
    """
    Check the convergence of the parameters based on R-hat and ESS values.
    """
    rhat_issues = summary_df[summary_df['R_hat'] > rhat_thresh]
    if not rhat_issues.empty:
        print(f"\nParameters with R-hat > {rhat_thresh}:")
        print(rhat_issues[['R_hat']])
    else:
        print(f"\nAll parameters passed R-hat < {rhat_thresh}")

    ess_issues = summary_df[(summary_df['ESS_bulk'] < ess_thresh) | (summary_df['ESS_tail'] < ess_thresh)]
    if not ess_issues.empty:
        print(f"\nParameters with ESS < {ess_thresh}:")
        print(ess_issues[['ESS_bulk', 'ESS_tail']])
    else:
        print(f"\nAll parameters passed ESS > {ess_thresh}")