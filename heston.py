import numpy as np

from py_vollib.black_scholes.implied_volatility import implied_volatility
from scipy.integrate import quad 
from tqdm import tqdm 

class Heston : 
    theta : float
    kappa : float
    nu : float
    rho : float
    r : float

    def __init__(self, theta, kappa, nu, rho, r) : 
        self.theta = theta
        self.kappa = kappa 
        self.nu = nu 
        self.rho = rho 
        self.r = r

    def feller(self) : 
        print("Feller condition ? ", 2 * self.theta * self.kappa," > ", self.nu**2, ".")
        return 

    def beta(self, v) : 
        return self.kappa - 1j * v * self.rho * self.nu

    def D(self, v) : 
        return np.sqrt(self.beta(v)**2 + self.nu**2 * v * (1j + v))

    def G(self, v) : 
        return (self.beta(v) - self.D(v)) / (self.beta(v) + self.D(v)) 

    def phi(self, v, t) : 
        return self.theta * self.kappa / self.nu**2 * ((self.beta(v) - self.D(v)) * t - 2 * np.log((self.G(v)*np.exp(-self.D(v)*t)-1)/(self.G(v)-1)))
    
    def psi(self, v, t) : 
        return (self.beta(v) - self.D(v)) / self.nu**2 * (1 - np.exp(-self.D(v)*t)) / (1 - self.G(v)*np.exp(-self.D(v)*t))

    def fonction_caracteristique(self, v, T, S, V, t = 0) : 
        return np.exp(1j * v * np.log(S) + 1j * v * self.r * (T-t) + self.phi(v, T-t) + self.psi(v, T-t) * V)

    def pi_1(self, T, K, S, V, t=0) :
        k = np.log(K)
        integrande = lambda z : (np.exp(-1j * z * k) / (1j * z) * self.fonction_caracteristique(z-1j, T, S, V, t)).real 
        integrale, _ = quad(integrande, 0, np.inf)
        return 0.5 + 1/np.pi * integrale * np.exp(-self.r *(T-t)) / S 
        
    def pi_2(self, T, K, S, V, t=0) :  
        k = np.log(K) 
        integrande = lambda z : (np.exp(-1j * z * k) * self.fonction_caracteristique(z, T, S, V, t) / (1j * z)).real
        integrale, _ = quad(integrande, 0, np.inf)
        return 0.5 + 1/np.pi * integrale 

    def call(self, T, K, S, V, t=0) : 
        return S * self.pi_1(T, K, S, V, t) - K * np.exp(-self.r * (T-t)) * self.pi_2(T, K, S, V, t)

    def put(self, T, K, S, V, t=0) :
        return self.call(T, K, S, V, t) - S + K * np.exp(-self.r * (T-t))

def generate_IV_Heston(T, K_range, S0, V0, r, kappa, theta, nu, rho) : 
    IV = np.zeros(len(K_range))
    model = Heston(theta, kappa, nu, rho, r)
    for i, K in enumerate(tqdm(K_range)) : 
        IV[i] = implied_volatility(model.call(T, K * np.exp(r * T), S0, V0), S0, K * np.exp(r * T), T, r, "c") 
    return IV

def generate_OTM_prices_Heston(T, K_range, S0, V0, r, kappa, theta, nu, rho) : 
    prices = np.zeros(len(K_range))
    model = Heston(theta, kappa, nu, rho, r)
    for i, K in enumerate(tqdm(K_range)) : 
        if K * np.exp(r * T) >= S0 : 
            prices[i] = model.call(T, K * np.exp(r * T), S0, V0)
        else : 
            prices[i] = model.put(T, K * np.exp(r * T), S0, V0)
    return prices 