import numpy as np 
import sys 

from scipy.optimize import minimize, least_squares
from tqdm import tqdm 

def svi_raw(k, a, b, m, rho, sigma) : 
    assert b >= 0, "b has to be non-negative"
    assert np.abs(rho) <= 1, "|rho| has to be smaller than 1"
    assert sigma >= 0, "sigma has to be non-negative"
    assert a + b * sigma * np.sqrt(1 - rho ** 2) >= 0, "a + b sigma (1-rho^2)^0.5 has to be non-negative"

    return a + b * (rho * (k - m) + np.sqrt(np.power(k - m, 2) + np.power(sigma, 2)))

def svi_jw(k, v, psi, p, c, hat_v, T) : 
    a, b, m, rho, sigma = from_jv_to_raw_parametrization(v, psi, p, c, hat_v, T)
    return svi_raw(k, a, b, m, rho, sigma) 

def from_raw_to_jw_parametrization(a, b, m, rho, sigma, T) : 
    v = (a + b * (-rho * m + np.sqrt(m**2 + sigma**2))) / T 
    w = v * T 
    psi = 1 / np.sqrt(w) * b / 2 * (rho - m / np.sqrt(m**2 + sigma**2))
    p = 1 / np.sqrt(w) * b * (1 - rho)
    c = 1 / np.sqrt(w) * b * (1 + rho)
    hat_v = 1 / T * (a + b * sigma * np.sqrt(1 - rho**2))
    return v, psi, p, c, hat_v 
    
def from_jv_to_raw_parametrization(v, psi, p, c, hat_v, T) : 
    w = v * T 
    b = np.sqrt(w) / 2 * (c + p)
    rho = 1 - p * np.sqrt(w) / b 
    
    beta = rho - 2 * psi * np.sqrt(w) / b
    if beta > 0 : 
        alpha = np.sqrt(1 / beta**2 - 1)
    elif beta < 0 : 
        alpha = -np.sqrt(1 / beta**2 - 1)
    else : 
        alpha = 0 
    if alpha > 0 : 
        m = (v - hat_v) * T / (b * (-rho + np.sqrt(1 + alpha**2) - alpha * np.sqrt(1 - rho**2)))
    elif alpha < 0 : 
        m = (v - hat_v) * T / (b * (-rho - np.sqrt(1 + alpha**2) - alpha * np.sqrt(1 - rho**2)))
    else : 
        m = (v - hat_v) * T / (b * (-rho - alpha * np.sqrt(1 - rho**2)))

    sigma = alpha * m if np.abs(m) > 1e-6 else (v * T - a) / b 
    a = hat_v * T - b * sigma * np.sqrt(1 - rho**2)

    return a, b, m, rho, sigma 

def durrleman_condition(k, a, b, m, rho, sigma) : 
    w = svi_raw(k, a, b, m, rho, sigma)
    partial_w = b * rho + b * (k - m) / np.sqrt((k - m)**2 + sigma**2)
    partial_2_w = b * sigma**2 / ((k - m)**2 + sigma**2)**(1.5)
    return (1 - 0.5 * k * partial_w / w)**2 - partial_w**2 / 4 * (1 / w + 0.25) + partial_2_w**2 /2 

def naive_calibration_SVI(k, v, T, M = 50) : 
    lower_bounds = np.array([0, 0, k.min(), -1, 0.001])
    upper_bounds = np.array([v.max(), 1, k.max(), 1, 1])

    vector = np.zeros((M, 6))

    def to_minimize(theta) : 
        a, b, m, rho, sigma = theta
        if b > 4 / (T * (1 + np.abs(rho))) :
            return 1e6
        vi = svi_raw(k, a, b, m, rho, sigma)
        return np.sum((vi - v)**2) 

    for i in tqdm(range(M), desc="Calibration"): 
        guess_init = np.array([
            v.min(),                                       # a 
            np.random.uniform(0, 1),                       # b
            np.random.uniform(k.min(), k.max()),           # m 
            np.random.uniform(-1, 1),                      # rho  
            np.random.uniform(0.001, 1)                    # sigma 
        ])

        res = least_squares(
            to_minimize,
            x0 = guess_init,
            bounds = (lower_bounds, upper_bounds),
            method="trf", 
            max_nfev=500
        )

        vector[i,:-1] = res.x
        vector[i, -1] = res.fun[0]

    min_idx = np.argmin(vector[:, -1])
    return vector[min_idx]    

class SVI :
    def __init__(self, k, var, vega, T):
        self.k = k
        self.n = len(k)
        self.T = T
        self.v = var 
        self.tilde_v = self.T * var
        self.max_tilde_v = np.max(self.tilde_v)
        self.vega = vega / vega.max() 
        self.omega = self.vega.sum() 

    def loss(self, c, d, a, y) : 
        diff = (self.tilde_v - (a + d * y + c * np.sqrt(y * y + 1)))
        return np.sum(self.vega * diff * diff)

    def check_in_D(self, c, d, a, sigma) : 
        cond1 = 0 <= c <= 4 * sigma 
        cond2 = np.abs(d) <= np.minimum(c, 4 * sigma - c)
        cond3 = 0 <= a <= self.max_tilde_v
        return cond1 and cond2 and cond3

    def to_minimize(self, theta) : 
        m, sigma = theta 

        c, d, tilde_a, loss = self.pre_calibration(m, sigma)
        if c != 0 : 
            a, b, rho = tilde_a / self.T, c / (sigma * self.T), d / c 
        else : 
            a, b, rho = tilde_a / self.T, 0, 0 
        
        try :
            vi = svi_raw(self.k, a, b, m, rho, sigma) 
            return np.sum(self.vega * (self.v - vi)**2)
        except AssertionError :
            return 1e12
        except Exception : 
            return 1e12  
    
    def to_minimize2(self, theta, v, psi, p, c, hat_v):
        c_bar, hat_v_bar = theta 
        a, b, m, rho, sigma = from_jv_to_raw_parametrization(v, psi, p, c_bar, hat_v_bar, self.T)
        if not (durrleman_condition(self.k, a, b, m, rho, sigma) >= 0).all() : 
            return 1e6 
        return np.sum(self.vega * (svi_jw(self.k, v, psi, p, c, hat_v, self.T) - svi_jw(self.k, v, psi, p, c_bar, hat_v_bar, self.T))**2)
    
    def calibration(self, m = None, sigma = None, lower_sigma = 0.005, rounded = True, without_butterfly_arbitrage = False) : 
        if m is None :  
            m = np.random.uniform(2 * (self.k).min(), 2 * (self.k).max())

        if sigma is None : 
            sigma = np.random.uniform(lower_sigma, 1)

        res = minimize(
            self.to_minimize,
            x0 = np.array([m, sigma]), 
            method = "Nelder-Mead",
            bounds = np.array([(None, None), (lower_sigma, 10)]),
            tol = 1e-16
        )

        m, sigma = res.x 
        c, d, tilde_a, loss = self.pre_calibration(m, sigma)
        if c != 0 : 
            a, b, rho = tilde_a / self.T, c / (sigma * self.T), d / c 
        else : 
            a, b, rho = tilde_a / self.T, 0, 0 
        
        if without_butterfly_arbitrage : 
            if not (durrleman_condition(self.k, a, b, m, rho, sigma) >= 0).all() : 
                a, b, m, rho, sigma = self.remove_butterfly_arbitrage(a, b, m, rho, sigma) 
        
        if rounded :
            return np.round([a, b, m, rho, sigma], 5) 
        return a, b, m, rho, sigma  

    def remove_butterfly_arbitrage(self, a, b, m, rho, sigma) : 
        v_star, psi_star, p_star, c_star, hat_v_star = from_raw_to_jw_parametrization(a, b, m, rho, sigma, self.T)
        c_tilde = p_star + 2 * psi_star 
        hat_v_tilde = v_star * 4 * p_star * c_tilde / (p_star + c_tilde)**2 

        res = minimize(
            self.to_minimize2,
            args = (v_star, psi_star, p_star, c_star, hat_v_star), 
            x0 = np.array([0.5 * (c_star + c_tilde), 0.5 * (hat_v_star + hat_v_tilde)]), 
            method = "Nelder-Mead",
            bounds = np.array([(np.minimum(c_star, c_tilde), np.maximum(c_star, c_tilde)), (np.minimum(hat_v_star, hat_v_tilde), np.maximum(hat_v_star, hat_v_tilde))]),
            tol = 1e-16
        )
        c_bar_opti, hat_v_bar_opti = res.x
        return from_jv_to_raw_parametrization(v_star, psi_star, p_star, c_bar_opti, hat_v_bar_opti, self.T)

    def MSE(self, a, b, m, rho, sigma) : 
        vi = svi_raw(self.k, a, b, m, rho, sigma)
        return np.sum((self.v - vi)**2)

    def pre_calibration(self, m, sigma) : 
        y = (self.k - m ) / sigma 
        y1 = np.sum(self.vega * y)
        y2 = np.sum(self.vega * y * y)
        y3 = np.sum(self.vega * np.sqrt(y * y + 1))
        y4 = np.sum(self.vega * y * np.sqrt(y * y + 1))
        y5 = np.sum(self.vega * (y * y + 1))

        vy2 = np.sum(self.vega * self.tilde_v * np.sqrt(y * y + 1))
        vy = np.sum(self.vega * self.tilde_v * y)
        v = np.sum(self.vega * self.tilde_v)

        A = np.array([
            [y5, y4, y3], 
            [y4, y2, y1],
            [y3, y1, self.omega]
        ])
        b = np.array([vy2, vy, v])
        c, d, tilde_a = np.linalg.solve(A, b)

        if self.check_in_D(c, d, tilde_a, sigma) : 
            loss = self.loss(c, d, tilde_a, y)
            return c, d, tilde_a, loss

        vector = []

        ## -------------- ## Faces (6) ## -------------- ## 

        # tilde_a = 0   
        A = np.array([
            [y5, y4, y3], 
            [y4, y2, y1],
            [0, 0, 1]
        ])
        b = np.array([vy2, vy, 0])
        c, d, tilde_a = np.linalg.solve(A, b)
        if self.check_in_D(c, d, tilde_a, sigma) : 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # tilde_a = self.max_tilde_v 
        A = np.array([
            [y5, y4], 
            [y4, y2]
        ])
        b = np.array([vy2 - self.max_tilde_v * y3, vy - self.max_tilde_v * y1])
        c, d = np.linalg.solve(A, b)
        tilde_a = self.max_tilde_v 
        if self.check_in_D(c, d, tilde_a, sigma) : 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c, 0 <= c <= 2 * sigma 
        A = np.array([
            [y2 + y5 + 2 * y4, y1 + y3],
            [y1 + y3, self.omega]
        ])
        b = np.array([vy2 + vy, v])
        c, tilde_a = np.linalg.solve(A, b)
        d = c 
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma: 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = -c, 0 <= c <= 2 * sigma 
        A = np.array([
            [y2 + y5 - 2 * y4, -y1 + y3],
            [-y1 + y3, self.omega]
        ])
        b = np.array([vy2 - vy, v])
        c, tilde_a = np.linalg.solve(A, b)
        d = -c
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma: 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = 4 * sigma - c, 2 * sigma <= c <= 4 * sigma
        A = np.array([
            [y2 + y5 - 2 * y4, -y1 + y3],
            [-y1 + y3, self.omega]
        ])
        b = np.array([vy2 - vy - 4 * sigma * (y4 - y2), v - 4 * sigma * y1])
        c, tilde_a = np.linalg.solve(A, b)
        d = 4 * sigma - c 
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma: 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c - 4 * sigma , 2 * sigma <= c <= 4 * sigma
        A = np.array([
            [y2 + y5 + 2 * y4, y1 + y3],
            [y1 + y3, self.omega]
        ])
        b = np.array([vy2 + vy  + 4 * sigma * (y4 + y2), v + 4 * sigma * y1])
        c, tilde_a = np.linalg.solve(A, b)
        d = c - 4 * sigma 
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma: 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # Note: c = 0, c = 4 * sigma implies that d = 0

        ## -------------- ## Edges (12) ## -------------- ## 

        # c = 0, d = 0
        tilde_a = v / self.omega
        c, d = 0, 0 
        if self.check_in_D(c, d, tilde_a, sigma):
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # c = 4 * sigma, d = 0 
        tilde_a = (v - 4 * sigma * y3) / self.omega 
        c, d = 4 * sigma, 0 
        if self.check_in_D(c, d, tilde_a, sigma):
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c, tilde_a = 0 
        c = (vy2 + vy) / (y5 + y2 + 2 * y4)
        d, tilde_a = c, 0 
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = -c, tilde_a = 0 
        c = (vy2 - vy) / (y5 + y2 - 2 * y4)
        d, tilde_a = -c, 0 
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = 4 * sigma - c, tilde_a = 0 
        c = (vy2 - vy - 4 * sigma * y4 + 4 * sigma * y2) / (y5 + y2 - 2 * y4)
        d, tilde_a = 4 * sigma - c, 0 
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c - 4 * sigma, tilde_a = 0 
        c = (vy2 + vy + 4 * sigma * y4 + 4 * sigma * y2) / (y5 + y2 + 2 * y4)
        d, tilde_a = c - 4 * sigma, 0  
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c, tilde_a = self.max_tilde_v 
        c = (vy2 + vy - self.max_tilde_v * (y3 + y1)) / (y5 + y2 + 2 * y4)
        d, tilde_a = c, self.max_tilde_v  
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = -c, tilde_a = self.max_tilde_v 
        c = (vy2 - vy - self.max_tilde_v * (y3 - y1)) / (y5 + y2 - 2 * y4)
        d, tilde_a = -c, self.max_tilde_v  
        if self.check_in_D(c, d, tilde_a, sigma) and 0 <= c <= 2 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = 4 * sigma - c, tilde_a = self.max_tilde_v 
        c = (vy2 - vy - 4 * sigma * y4 + 4 * sigma * y2 - self.max_tilde_v * (y3 - y1)) / (y5 + y2 - 2 * y4)
        d, tilde_a = 4 * sigma - c, self.max_tilde_v
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # d = c - 4 * sigma, tilde_a = self.max_tilde_v 
        c = (vy2 + vy + 4 * sigma * y4 + 4 * sigma * y2 - self.max_tilde_v * (y3 + y1)) / (y5 + y2 + 2 * y4)
        d, tilde_a = c - 4 * sigma, self.max_tilde_v 
        if self.check_in_D(c, d, tilde_a, sigma) and 2 * sigma <= c <= 4 * sigma :
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # c = 2 * sigma, d = 2 * sigma 
        tilde_a = (v - 2 * sigma * (y3 + y1)) / self.omega
        c, d = 2 * sigma, 2 * sigma 
        if self.check_in_D(c, d, tilde_a, sigma) : 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])

        # c = 2 * sigma, d = -2 * sigma 
        tilde_a = (v - 2 * sigma * (y3 - y1)) / self.omega
        c, d = 2 * sigma, -2 * sigma 
        if self.check_in_D(c, d, tilde_a, sigma) : 
            loss = self.loss(c, d, tilde_a, y)
            vector.append([c, d, tilde_a, loss])
    
        ## -------------- ## Corners (8) ## -------------- ##  
         
        # c = 0, d = 0, tilde_a = 0 
        loss = self.loss(0, 0, 0, y)
        vector.append([0, 0, 0, loss])

        # c = 0, d = 0, tilde_a = self.max_tilde_v 
        loss = self.loss(0, 0, self.max_tilde_v, y)
        vector.append([0, 0, self.max_tilde_v, loss])

        # c = 2 * sigma, d = 2 * sigma, tilde_a = 0 
        loss = self.loss(2 * sigma, 2 * sigma, 0, y)
        vector.append([2 * sigma, 2 * sigma, 0, loss])

        # c = 2 * sigma, d = 2 * sigma, tilde_a = self.max_tilde_v 
        loss = self.loss(2 * sigma, 2 * sigma, self.max_tilde_v, y)
        vector.append([2 * sigma, 2 * sigma, self.max_tilde_v, loss])

        # c = 2 * sigma, d = -2 * sigma, tilde_a = 0 
        loss = self.loss(2 * sigma, -2 * sigma, 0, y)
        vector.append([2 * sigma, -2 * sigma, 0, loss])

        # c = 2 * sigma, d = -2 * sigma, tilde_a = self.max_tilde_v 
        loss = self.loss(2 * sigma, -2 * sigma, self.max_tilde_v, y)
        vector.append([2 * sigma, -2 * sigma, self.max_tilde_v, loss])

        # c = 4 * sigma, d = 0, tilde_a = 0 
        loss = self.loss(4 * sigma, 0, 0, y)
        vector.append([4 * sigma, 0, 0, loss])

        # c = 4 * sigma, d = 0, tilde_a = self.max_tilde_v 
        loss = self.loss(4 * sigma, 0, self.max_tilde_v, y)
        vector.append([4 * sigma, 0, self.max_tilde_v, loss])


        vector = np.array(vector) 
        min_idx = np.argmin(vector[:, -1])
        return vector[min_idx]

class MultiSlicesSVI : 
    def __init__(self, k, var_values, vega_values, T_values) : 
        self.k = k 
        self.n = len(self.k)
        self.var_values = var_values 
        self.T_values = T_values
        self.vega_values = vega_values 

    def to_minimize(self, theta, i, without_butterfly_arbitrage) : 
        a_1, b_1, m_1, rho_1, sigma_1, a_2, b_2, m_2, rho_2, sigma_2 = theta

        if without_butterfly_arbitrage : 
            if not (durrleman_condition(self.k, a_1, b_1, m_1, rho_1, sigma_1) >= 0).all() : 
                return 1e12 
            if not (durrleman_condition(self.k, a_2, b_2, m_2, rho_2, sigma_2) >= 0).all() : 
                return 1e12 

        res = 0 
        try :
            vi_1 = svi_raw(self.k, a_1, b_1, m_1, rho_1, sigma_1)
            vi_2 = svi_raw(self.k, a_2, b_2, m_2, rho_2, sigma_2)
            res += np.sum((vi_1 - self.var_values[i])**2) + np.sum((vi_2 - self.var_values[i+1])**2) 
        except AssertionError :
            return 1e12
        except Exception : 
            return 1e12 

        k_values = np.zeros(self.n + 1)
        k_values[0] = self.k[0] - 1
        k_values[1:-1] = 0.5 * (self.k[1:] + self.k[:-1]) 
        k_values[-1] = self.k[-1] + 1 
    
        c = np.maximum(self.T_values[i] * svi_raw(k_values, a_1, b_1, m_1, rho_1, sigma_1)  - self.T_values[i+1] * svi_raw(k_values, a_2, b_2, m_2, rho_2, sigma_2), 0.0) 
        res += np.sum(c**2) * 1e6
        return res 

    def calibration(self, lower_sigma = 0.005, rounded = False, without_butterfly_arbitrage = True, without_calendar_arbitrage = False) : 
        result = np.zeros((len(self.T_values), 6))
        for i, T in enumerate(self.T_values) : 
            svi = SVI(self.k, self.var_values[i], self.vega_values[i], T)
            a, b, m, rho, sigma = svi.calibration(lower_sigma = lower_sigma, rounded = rounded, without_butterfly_arbitrage = without_butterfly_arbitrage)
            result[i,:-1] = [a, b, m, rho, sigma]
            result[i, -1] = svi.MSE(a, b, m, rho, sigma)

        if without_calendar_arbitrage :
            for i in range(len(self.T_values)-1) : 
                a_1, b_1, m_1, rho_1, sigma_1 = result[i, :-1]
                a_2, b_2, m_2, rho_2, sigma_2 = result[i+1, :-1]
                if (self.T_values[i] * svi_raw(self.k, a_1, b_1, m_1, rho_1, sigma_1) - self.T_values[i+1] * svi_raw(self.k, a_2, b_2, m_2, rho_2, sigma_2) < 0).all() : 
                    continue 
                res = minimize(
                    self.to_minimize, 
                    args = (i, without_butterfly_arbitrage),
                    x0 = np.array([
                        a_1, b_1, m_1, rho_1, sigma_1,
                        a_2, b_2, m_2, rho_2, sigma_2]),
                    bounds = np.array([
                        (0, max(self.var_values[i])),
                        (0, 1), 
                        (2 * min(self.k), 2 * max(self.k)),
                        (-1, 1), 
                        (0.005, 1),
                        (0, max(self.var_values[i+1])),
                        (0, 1), 
                        (2 * min(self.k), 2 * max(self.k)),
                        (-1, 1), 
                        (0.005, 1)]),
                    method = "Nelder-Mead", # SLSQP should be more suitable for high-dimensionality problems 
                    tol = 1e-16 
                )
                a_1, b_1, m_1, rho_1, sigma_1, a_2, b_2, m_2, rho_2, sigma_2 = res.x 
                result[i, :-1] = a_1, b_1, m_1, rho_1, sigma_1
                result[i+1, :-1] = a_2, b_2, m_2, rho_2, sigma_2
                # If you want to update the MSE, you just need to keep the class instances in memory  
        return result 
