#include "scran.h"

struct truncnorm {
    truncnorm(size_t, double, double, double, double);
    double sum, sqsum, left, right;
    size_t n;
    
    void compute(double, double);
    double loglik, dmu, dsig, ddmusig, ddmu2, ddsig2;
};

truncnorm::truncnorm(size_t N, double Sum, double Sqsum, double Left, double Right) : 
    n(N), sum(Sum), sqsum(Sqsum), left(Left), right(Right) {}

void truncnorm::compute(double mu, double sigma) {
    /* Log-likelihood for a truncated normal at [left, right]:
     *
     * SUM[
     *     - 0.5 * log(2*pi) - log(sigma)
     *     - 0.5 ( (x - mu)/sigma )^2
     *     - log( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     * ]
     * 
     * ... which, for the given sufficient statistics, is equal to:
     *
     * n * [ 
     *     - 0.5 * log(2*pi) - log(sigma) 
     *     - log( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     * ] - 0.5 / sigma^2 * [
     *     SUM(x^2) - 2 * SUM(x) * mu + n * mu ^ 2
     * ]
     */ 
    const double denom = R::pnorm(right, mu, sigma, 1, 0) - R::pnorm(left, mu, sigma, 1, 0);
    const double sigma2 = sigma*sigma;
    const double inner_sum = sqsum - 2 * sum * mu + n * mu * mu;
    loglik = 
        n * (
            - M_LN_SQRT_2PI - std::log(sigma)
            - std::log(denom)
        ) - 0.5 / sigma2 * (
            inner_sum
        );

    /* First derivative for truncated normal log-likelihood, with respect to mu:
     * 
     * n * [
     *     ( dnorm(right, mu, sigma) - dnorm(left, mu, sigma) )
     *         / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     * ] - 1 / sigma^2 * [
     *     - SUM(X) + n * mu
     * ]
     * 
     * Note that the derivative of pnorm w.r.t. the mean is simply the normal PDF with a flipped sign.
     */
    const double right_dnorm=R::dnorm(right, mu, sigma, 0), left_dnorm=R::dnorm(left, mu, sigma, 0);
    const double diff_dnorm = (right_dnorm - left_dnorm)/denom;
    dmu =
        n * (
            diff_dnorm 
        ) - 1 / sigma2 * (
            - sum + n * mu
        );

    /* First derivative with respect to sigma:
     * 
     * n * [
     *    - 1 / sigma 
     *    + 1 / sigma * (
     *        (right - mu) * dnorm(right, mu, sigma) 
     *        - (left - mu) * dnorm(left, mu, sigma) 
     *    ) / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     * ] + 1 / sigma^3 * [
     *    SUM(x^2) - 2 * SUM(x) * mu + n * mu ^ 2
     * ]
     */
    const double sigma3 = sigma2 * sigma;
    const double right_gap = (right - mu), left_gap = (left - mu);
    const double diff_gap_dnorm = (right_gap * right_dnorm - left_gap * left_dnorm)/ denom;
    
    dsig =
        n /sigma  * (
            - 1 + diff_gap_dnorm
        ) + 1 / sigma3 * (
            inner_sum
        );
            
    /* Second derivative for mu and mu:
     * 
     * n * [
     *     ( dnorm(right, mu, sigma) - dnorm(left, mu, sigma) )^2
     *         / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) ) ^2
     *     + ( 
     *         dnorm(right, mu, sigma) * (right - mu)/sigma^2
     *         - dnorm(left, mu, sigma) * (left - mu)/sigma^2
     *     ) / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     * ] - n / sigma^2
     * 
     */
    ddmu2 = 
        n * (
            diff_dnorm * diff_dnorm 
            + diff_gap_dnorm / sigma2 
            - 1/sigma2
        );

    /* Second derivative for sigma and sigma:
     * n * [
     *    1 / sigma^2 
     *    - 1 / sigma^2 * (
     *        (right - mu) * dnorm(right, mu, sigma) 
     *        - (left - mu) * dnorm(left, mu, sigma) 
     *    ) / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     *    + 1 / sigma * (
     *        - 1 / sigma * ( 
     *            (right - mu) * dnorm(right, mu, sigma) 
     *            - (left - mu) * dnorm(left, mu, sigma) )
     *        + 1 / sigma^3 * ( 
     *            (right - mu)^3 * dnorm(right, mu, sigma)
     *            - (left - mu)^3 * dnorm(left, mu, sigma)
     *        )
     *    ) / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     *    + 1 / sigma * (
     *        (right - mu) * dnorm(right, mu, sigma) 
     *        - (left - mu) * dnorm(left, mu, sigma) 
     *    )^2 / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) ) ^2
     * ] - 3 / sigma^4 * [
     *    SUM(x^2) - 2 * SUM(x) * mu + n * mu ^ 2
     * ]
     * 
     * Note the simplification of terms to get the "2 * diff_gap_dnorm/sigma2" below.
     */
    ddsig2 = 
        n * (
            1 / sigma2
            - 2 * diff_gap_dnorm / sigma2
            + 1 / (sigma2 * sigma2) * (
                right_gap * right_gap * right_gap * right_dnorm 
                - left_gap * left_gap * left_gap * left_dnorm 
            ) / denom 
            + 1 / sigma2 * diff_gap_dnorm * diff_gap_dnorm 
        ) - 3 / (sigma2 * sigma2) * inner_sum;

    /* Second derivative for mu, then sigma:
     *
     * n * [
     *     ( 
     *        - 1 / sigma * ( 
     *            dnorm(right, mu, sigma) 
     *            - dnorm(left, mu, sigma) 
     *        ) + 1 / sigma^3 * ( 
     *            (right - mu)^2 * dnorm(right, mu, sigma)
     *            - (left - mu)^2 * dnorm(left, mu, sigma)
     *        )
     *     ) / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) )
     *     + 1 / sigma * (
     *         (right - mu) * dnorm(right, mu, sigma) 
     *         - (left - mu) * dnorm(left, mu, sigma) 
     *     ) * ( dnorm(right, mu, sigma) - dnorm(left, mu, sigma) )
     *         / ( pnorm(right, mu, sigma) - pnorm(left, mu, sigma) ) ^ 2
     * ] + 2 / sigma^3 * [
     *     - SUM(X) + n * mu
     * ]
     */
    ddmusig = 
        n * (
            - 1 / sigma * diff_dnorm
            + 1 / sigma3 * (
                right_gap * right_gap * right_dnorm 
                - left_gap * left_gap * left_dnorm 
            ) / denom 
            + 1 / sigma * diff_gap_dnorm * diff_dnorm
        ) + 2 / sigma3 * (
            - sum + n * mu
        );

    // Second derivative for sigma, then mu: this is the same as ddmusig.

    return;    
}

SEXP truncnorm_test(SEXP N, SEXP Sum, SEXP Sumsq, SEXP Left, SEXP Right, SEXP Mean, SEXP Sigma) {
    BEGIN_RCPP
    truncnorm TN(
        check_integer_scalar(N, "number of observations"),
        check_numeric_scalar(Sum, "sum of observations"),
        check_numeric_scalar(Sumsq, "sum of squared observations"),
        check_numeric_scalar(Left, "left boundary"),
        check_numeric_scalar(Right, "right boundary")
    );
    TN.compute(
        check_numeric_scalar(Mean, "current mean"),
        check_numeric_scalar(Sigma, "current sigma")
    );

    return Rcpp::NumericVector::create(TN.loglik, TN.dmu, TN.dsig, TN.ddmu2, TN.ddsig2, TN.ddmusig);
    END_RCPP
}

/* Applying a multivariate Newton-Raphson fitter. Note the regularization. */

std::tuple<double, double, bool> fit_truncnorm(size_t n, double sum, double sumsq, double left, double right, double tol=0.00000001, size_t maxit=100) {
    double cur_mean=sum/n;
    double cur_sd=std::sqrt(sumsq/n - cur_mean * cur_mean);
    if (cur_mean <= 0 || cur_sd <= 0) {
        return std::tuple<double, double, bool>(std::max(0.0, cur_mean), std::max(0.0, cur_sd), false);
    }

    truncnorm current(n, sum, sumsq, left, right);
    truncnorm proposed(current);
    proposed.compute(cur_mean, cur_sd);

    size_t it=0;
    while ((++it) <= maxit) {
        current=proposed;

        // Inverting the 2 x 2 Hessian and multiplying by the first derivatives to get the steps.
        const double det = 1/(current.ddmu2 * current.ddsig2 - current.ddmusig * current.ddmusig); 
        double step_mean = - det * (current.ddsig2 * current.dmu - current.ddmusig * current.dsig);
        double step_sd = - det * (- current.ddmusig * current.dmu + current.ddmu2 * current.dsig);

        // Guarantee that the log-likelihood increases, and that the values are never negative.
        bool innerstep=false;
        while (innerstep = std::abs(step_mean/cur_mean) > tol) {
            double new_mean = cur_mean + step_mean;
            double new_sd = cur_sd + step_sd;

            if (new_mean > 0 && new_sd > 0) {
                proposed.compute(new_mean, new_sd);
                if (proposed.loglik > current.loglik) {
                    break;
                }
            }
             
            step_mean /= 10;
            step_sd /= 10;
        }

        if (!innerstep) {
            break;
        }
        cur_mean += step_mean;
        cur_sd += step_sd;
    }

    return std::tuple<double, double, bool>(cur_mean, cur_sd, it < maxit);
}

SEXP fit_truncnorm_test(SEXP N, SEXP Sum, SEXP Sumsq, SEXP Left, SEXP Right) {
    BEGIN_RCPP
    auto loc=fit_truncnorm(
        check_integer_scalar(N, "number of observations"),
        check_numeric_scalar(Sum, "sum of observations"),
        check_numeric_scalar(Sumsq, "sum of squared observations"),
        check_numeric_scalar(Left, "left boundary"),
        check_numeric_scalar(Right, "right boundary")
    );
    return Rcpp::NumericVector::create(std::get<0>(loc), std::get<1>(loc));
    END_RCPP
}

