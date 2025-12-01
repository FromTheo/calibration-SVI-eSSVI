import numpy as np 

from scipy.optimize import brent, least_squares
from scipy.stats import norm 
from tqdm import tqdm

def tvi_raw(k, theta, rho, phi) : 
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + 1 - rho**2))

def BS_normalized_price(k, sigma, T, option_type="call"):
    k = np.asarray(k)
    d1 = (-k + 0.5 * sigma**2 * T) / sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    else:
        return np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)

def BS_OTM_price(k, sigma, T) :
    k = np.asarray(k)
    prices = np.empty_like(k, dtype=float)
    mask_put = (k < 0)
    prices[mask_put] = BS_normalized_price(k[mask_put], sigma[mask_put], T, "put")
    mask_call = ~mask_put
    prices[mask_call] = BS_normalized_price(k[mask_call], sigma[mask_call], T, "call")
    return prices 

def from_ssvi_to_svi_parametrization(theta, rho, phi) : 
    a = theta * (1 - rho**2) / 2
    b = theta * phi / 2 
    m = -rho / phi 
    sigma = np.sqrt(1 - rho**2) / phi 
    return a, b, m, rho, sigma 

class eSSVI : 
    def __init__(self, k_values, tvi_market, T_values, k_star_values, theta_star_values) : 
        self.k_values = k_values 
        self.tvi_market = tvi_market 
        self.T_values = T_values 
        self.k_star_values = k_star_values
        self.theta_star_values = theta_star_values 

    def psi_plus(self, i, rho) : 
        return -2 * rho * self.k_star_values[i] / (1 + np.abs(rho)) + np.sqrt(4 * rho**2 * self.k_star_values[i]**2 / (1 + np.abs(rho))**2 + 4 * self.theta_star_values[i] / (1 + np.abs(rho)))
    
    def psi_minus(self, i, rho, previous_psi, previous_rhopsi) : 
        return np.maximum((previous_psi - previous_rhopsi) / (1 - rho), (previous_psi + previous_rhopsi) / (1 + rho))

    def loss(self, psi, i, rho) :
        theta = self.theta_star_values[i] - rho * psi * self.k_star_values[i]
        phi = psi / theta 
        tvi_model = tvi_raw(self.k_values[i], theta, rho, phi) 
        return np.mean(np.abs(tvi_model - self.tvi_market[i]))

    def rho_psi_butterfly_only(self, i, M, eps=1e-8, tol=1e-6, max_outer_iter=15):
        rho_best = 0.0
        psi_best = None
        loss_best = np.inf

        center = 0.0
        width = 1.0

        it = 0
        while it < max_outer_iter:
            rho_min = np.maximum(center - width, -1 + eps)
            rho_max = np.minimum(center + width,  1 - eps)
            rho_values = np.random.uniform(rho_min, rho_max, M)

            local_loss_best = np.inf
            local_rho_best = None
            local_psi_best = None

            for j in range(M):
                rho = rho_values[j]
                a = eps
                b = np.minimum(self.psi_plus(i, rho) - eps, 4 / (1 + np.abs(rho)) - eps)
                psi = brent(self.loss, args=(i, rho), brack=(a, b))
                loss_candid = self.loss(psi, i, rho)
                if loss_candid < local_loss_best:
                    local_loss_best = loss_candid
                    local_rho_best = rho
                    local_psi_best = psi
                if loss_candid < loss_best:
                    loss_best = loss_candid
                    rho_best = rho
                    psi_best = psi

            center = local_rho_best if local_rho_best is not None else center
            width /= 2.0
            it += 1

            if loss_best < tol:
                break

        return rho_best, psi_best

    def rho_psi(self, i, M, previous_theta, previous_psi, previous_rhopsi, eps=1e-8, tol=1e-6, max_outer_iter=15):
        rho_best = 0.0
        psi_best = None
        loss_best = np.inf

        center = 0.0
        width = 1.0

        it = 0
        while it < max_outer_iter:
            rho_min = np.maximum(center - width, -1 + eps)
            rho_max = np.minimum(center + width,  1 - eps)
            rho_values = np.random.uniform(rho_min, rho_max, M)

            local_loss_best = np.inf
            local_rho_best = None
            local_psi_best = None

            for j in range(M):
                rho = rho_values[j]
                k_star = self.k_star_values[i]
                theta_star = self.theta_star_values[i]

                if rho * k_star < 0:
                    a = np.maximum(
                        np.maximum(
                            eps,
                            np.maximum(
                                previous_psi + eps,
                                self.psi_minus(i, rho, previous_psi, previous_rhopsi)
                            )
                        ),
                        (theta_star - previous_theta) / (rho * k_star)
                    )
                    b = np.minimum(
                        self.psi_plus(i, rho),
                        4 / (1 + np.abs(rho)) - eps
                    )

                elif rho * k_star > 0:
                    a = np.maximum(
                        eps,
                        np.maximum(
                            previous_psi + eps,
                            self.psi_minus(i, rho, previous_psi, previous_rhopsi)
                        )
                    )
                    b = np.minimum(
                        np.minimum(
                            self.psi_plus(i, rho),
                            4 / (1 + np.abs(rho)) - eps
                        ),
                        (theta_star - previous_theta) / (rho * k_star)
                    )

                else: 
                    assert theta_star > previous_theta, "theta_star must be greater than previous_theta"
                    a = np.maximum(
                        eps,
                        np.maximum(
                            previous_psi + eps,
                            self.psi_minus(i, rho, previous_psi, previous_rhopsi)
                        )
                    )
                    b = np.minimum(
                        self.psi_plus(i, rho),
                        4 / (1 + np.abs(rho)) - eps
                    )

                psi = brent(self.loss, args=(i, rho), brack=(a, b))
                loss_candid = self.loss(psi, i, rho)

                if loss_candid < local_loss_best:
                    local_loss_best = loss_candid
                    local_rho_best = rho
                    local_psi_best = psi

                if loss_candid < loss_best:
                    loss_best = loss_candid
                    rho_best = rho
                    psi_best = psi

            if local_rho_best is None:
                break

            center = local_rho_best
            width /= 2.0
            it += 1

            if loss_best < tol:
                break

        return rho_best, psi_best

    def calibration(self, M=20, eps = 1e-8, tol = 1e-6) : 
        result = np.zeros((len(self.T_values), 3))
        for i in tqdm(range(len(self.T_values))) : 
            if i == 0 :
                rho, psi = self.rho_psi_butterfly_only(i, M) 
                theta = self.theta_star_values[i] - rho * psi * self.k_star_values[i]
                result[0, :] = [theta, rho, psi / theta]
            else : 
                previous_theta, previous_rho, previous_phi = result[i-1, :]
                previous_psi = previous_theta * previous_phi 
                previous_rhopsi = previous_rho * previous_psi 
                rho, psi = self.rho_psi(i, M, previous_theta, previous_psi, previous_rhopsi)
                theta = self.theta_star_values[i] - rho * psi * self.k_star_values[i]
                result[i, :] = [theta, rho, psi / theta]
        return result     

class eSSVI_2 : 
    def __init__(self, k_values, T_values, theta_star, tiv_market, prices_otm_market) :
        self.k_values = k_values 
        self.T_values = T_values 
        self.n = len(self.T_values)
        self.theta_star = theta_star  
        self.tiv_market = tiv_market 
        self.prices_otm_market = prices_otm_market   

    def f(self, theta, abs_rho) : 
        return 4 * theta / (1 + abs_rho) 

    def compute_w(self, params) : 
        rho_values = params[:self.n] 
        theta_1 = params[self.n]
        a_values = params[(self.n + 1):(2 * self.n)] 
        c_values = params[(2 * self.n):] 

        p_values = np.maximum((1 + rho_values[:-1]) / (1 + rho_values[1:]),
                              (1 - rho_values[:-1]) / (1 - rho_values[1:]))

        theta_values = np.zeros(self.n) 
        theta_values[0] = theta_1
        for i in range(1, self.n) : 
            theta_values[i] = theta_values[i-1] * p_values[i-1] + a_values[i-1]

        f_values = np.minimum(4 / (1 + np.abs(rho_values)), np.sqrt(self.f(theta_values, np.abs(rho_values))))

        cum_prod = np.concatenate(([1.0], np.cumprod(p_values)))
    
        C_psi_1 = np.min(f_values / cum_prod)
    
        psi_1 = c_values[0] * C_psi_1 
        psi_values = np.zeros(self.n)
        psi_values[0] = psi_1 

        A_psi_values, C_psi_values = np.zeros(self.n), np.zeros(self.n)
        C_psi_values[0] = C_psi_1 
    
        for i in range(1, self.n) : 
            A_psi_values[i] = psi_values[i-1] * p_values[i-1]
            cum_prod = np.concatenate(([1.0], np.cumprod(p_values[i:])))
            f_values_troncatured = f_values[i:]
            C_psi_values[i] = np.minimum(np.min(f_values_troncatured / cum_prod), 
                                         psi_values[i-1] / theta_values[i-1] * theta_values[i])
            psi_values[i] = c_values[i] * (C_psi_values[i] - A_psi_values[i]) + A_psi_values[i] 

        w_values = []
        for i in range(self.n) : 
            w = tvi_raw(self.k_values[i], theta_values[i], rho_values[i], psi_values[i] / theta_values[i])
            w_values.append(w)
        
        return w_values 
            
    
    def to_minimize(self, params) :
        rho_values = params[:self.n] 
        theta_1 = params[self.n]
        a_values = params[(self.n + 1):(2 * self.n)]
        c_values = params[(2 * self.n):] 

        p_values = np.maximum((1 + rho_values[:-1]) / (1 + rho_values[1:]),
                              (1 - rho_values[:-1]) / (1 - rho_values[1:])) 

        theta_values = np.zeros(self.n) 
        theta_values[0] = theta_1
        for i in range(1, self.n) : 
            theta_values[i] = theta_values[i-1] * p_values[i-1] + a_values[i-1]

        f_values = np.minimum(4 / (1 + np.abs(rho_values)), np.sqrt(self.f(theta_values, np.abs(rho_values))))

        cum_prod = np.concatenate(([1.0], np.cumprod(p_values)))
    
        C_psi_1 = np.min(f_values / cum_prod)
    
        psi_1 = c_values[0] * C_psi_1 
        psi_values = np.zeros(self.n)
        psi_values[0] = psi_1 

        A_psi_values, C_psi_values = np.zeros(self.n), np.zeros(self.n) 
        C_psi_values[0] = C_psi_1 
    
        for i in range(1, self.n) : 
            A_psi_values[i] = psi_values[i-1] * p_values[i-1]
            cum_prod = np.concatenate(([1.0], np.cumprod(p_values[i:])))
            f_values_troncatured = f_values[i:]
            C_psi_values[i] = np.minimum(np.min(f_values_troncatured / cum_prod), 
                                         psi_values[i-1] / theta_values[i-1] * theta_values[i])
            psi_values[i] = c_values[i] * (C_psi_values[i] - A_psi_values[i]) + A_psi_values[i] 

        residuals = []
        for i in range(self.n) : 
            w = tvi_raw(self.k_values[i], theta_values[i], rho_values[i], psi_values[i] / theta_values[i])
            residuals.append(w - self.tiv_market[i])
            """
            To be aligned with the paper: 
            prices_otm_model = BS_OTM_price(self.k_values[i], np.sqrt(w / self.T_values[i]), self.T_values[i])
            residuals.append(prices_otm_model - self.prices_otm_market[i]) 
            """

        residuals = np.concatenate(residuals)
        return residuals

    def calibration(self) :
        lower_rho = -np.ones(self.n)
        guess_rho = np.zeros(self.n)
        upper_rho = np.ones(self.n)

        lower_theta_1 = np.array([0])
        upper_theta_1 = np.array([np.inf])

        lower_a = np.zeros(self.n - 1)
        guess_a = self.theta_star[1:] - self.theta_star[:-1]
        upper_a = np.full(self.n - 1, np.inf)

        lower_c = np.zeros(self.n) 
        guess_c = np.ones(self.n) * 0.5
        upper_c = np.ones(self.n) 

        lower_bounds = np.concatenate([lower_rho, lower_theta_1, lower_a, lower_c], axis = 0)
        upper_bounds = np.concatenate([upper_rho, upper_theta_1, upper_a, upper_c], axis = 0)

        guess_init = np.concatenate([guess_rho, np.array([self.theta_star[0]]), guess_a, guess_c], axis = 0)

        res = least_squares(
            self.to_minimize,
            x0 = guess_init,
            bounds = (lower_bounds, upper_bounds)
        )

        return res 




        

        