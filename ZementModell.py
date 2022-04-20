# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:04:03 2022

@author: Klaus Götz
"""

import scipy.special
from scipy.integrate import romberg, nquad

import numpy as np
import matplotlib.pyplot as plt

def SchultzDistribution(n, mean, width):
    f_S = np.power((width+1/mean),width+1)*np.power(n,width)*np.exp(-1*n*(width+1)/(mean))/scipy.special.gamma(width+1)
    return f_S

def interglobuleSF(Q, fractal_dimension, fractal_cutoff, globule_radius):
    
    D = fractal_dimension
    xsi = fractal_cutoff
    Re = globule_radius    
    
    S_Q = (1 + np.power(xsi/Re, D)*scipy.special.gamma(D+1)*
           (np.sin((D-1)*np.arctan(Q*xsi)))/
           ((D-1)*np.power(1+np.power(Q*xsi,2),(D-1)/2)*Q*xsi))
    
    return S_Q

def particleSF_parts(Q, mu, cylinder_radius, water_height, csh_height, density_water,
               density_csh, density_solvent, layer_number):
    
    R = cylinder_radius
    L1 = csh_height
    L2 = water_height
    L = L1+L2
    
    n = layer_number
    
    rho_1 = density_csh
    rho_2 = density_water
    rho_s = density_solvent
    
    if mu == 0:
        A = np.zeros(len(Q))+((rho_1-rho_s)*L1/2 + (rho_2-rho_s)*L2/2)
        
        B = np.zeros(len(Q))
                
        C = np.zeros(len(Q))+( 2 / ((rho_1-rho_s)*L1+(rho_2-rho_s)*L2))
        
        D = (2*scipy.special.jv(1, Q*R) / (Q*R))
        
    elif mu == 1:
        A = ((rho_1-rho_s)*np.cos(Q*(n*L-L2)/2)*(np.sin(Q*L1/2))/(Q) +
             (rho_2-rho_s)*np.cos(Q*(n*L+L1)/2)*(np.sin(Q*L2/2))/(Q))
        
        B = ((rho_1-rho_s)*np.sin(Q*(n*L-L2)/2)*(np.sin(Q*L1/2))/(Q) +
             (rho_2-rho_s)*np.sin(Q*(n*L+L1)/2)*(np.sin(Q*L2/2))/(Q))

        C = ( 2 / (n*((rho_1-rho_s)*L1+(rho_2-rho_s)*L2)) *
                (np.sin(Q*n*L/2))/(np.sin(Q*L/2)))
            
        D1 = (Q*R*np.sqrt(1-np.power(mu,2)))
        D2 = (Q*R*np.sqrt(1-np.power(mu,2)))
        
        D = np.zeros(len(Q))+1
        
    else:
        A = ((rho_1-rho_s)*np.cos(Q*mu*(n*L-L2)/2)*(np.sin(Q*mu*L1/2))/(Q*mu) +
             (rho_2-rho_s)*np.cos(Q*mu*(n*L+L1)/2)*(np.sin(Q*mu*L2/2))/(Q*mu))
        
        B = ((rho_1-rho_s)*np.sin(Q*mu*(n*L-L2)/2)*(np.sin(Q*mu*L1/2))/(Q*mu) +
             (rho_2-rho_s)*np.sin(Q*mu*(n*L+L1)/2)*(np.sin(Q*mu*L2/2))/(Q*mu))

        C = ( 2 / (n*((rho_1-rho_s)*L1+(rho_2-rho_s)*L2)) *
                (np.sin(Q*mu*n*L/2))/(np.sin(Q*mu*L/2)))
            
        D1 = 2*scipy.special.jv(1, Q*R*np.sqrt(1-np.power(mu,2)))
        D2 = (Q*R*np.sqrt(1-np.power(mu,2)))
        
        D = D1/D2
        
    return A, B, C, D

def particleSF(Q, mu, cylinder_radius, water_height, csh_height, density_water,
               density_csh, density_solvent, layer_number):
    
    A, B, C, D = particleSF_parts(Q, mu, cylinder_radius, water_height,
                                  csh_height, density_water, density_csh,
                                  density_solvent, layer_number)
    P_Q = np.power(D,2)*np.power(C,2)*(np.power(A,2)+np.power(B,2))
    
    return P_Q

def average_orientation_PQ(Q, cylinder_radius, water_height, csh_height, density_water,
               density_csh, density_solvent, layer_number):
    if type(Q) != np.float64:
        P_Q = []
        
        for q in Q:
            integration_function = lambda x: particleSF(q, x, cylinder_radius, water_height,
                                   csh_height, density_water, density_csh, density_solvent, layer_number)
            integral = romberg(integration_function, 0, 1, rtol=1e-3)
            P_Q.append(integral)
        return np.array(P_Q)
    else:
        integration_function = lambda x: particleSF(Q, x, cylinder_radius, water_height,
                               csh_height, density_water, density_csh, density_solvent, layer_number)
        integral = romberg(integration_function, 0, 1, rtol=1e-3, divmax=20)
        return integral

def average_polydisperse_PQ(Q, cylinder_radius, water_height, csh_height, density_water,
               density_csh, density_solvent, layer_number_mean, layer_number_width):
    P_Q = []
    P_err = []
    
    for q in Q:
        print(q)
        integratation_function = lambda x,y : (particleSF(q, y, cylinder_radius,
                water_height, csh_height, density_water, density_csh, density_solvent, x) * 
                SchultzDistribution(x, layer_number_mean, layer_number_width))

        integral = nquad(integratation_function, [[0, np.inf], [0, 1]],
                         opts={'limit':100, 'epsrel':1e-3})
        P_Q.append(integral[0])
        P_err.append(integral[1])
    return np.array(P_Q), np.array(P_err)

def intensity(Q, cylinder_radius, water_height, csh_height, density_water, background,
               density_csh, density_solvent, layer_number_mean, layer_number_width,
               fractal_dimension, fractal_cutoff, globule_radius, number_density):

    L1 = csh_height
    L2 = water_height
    
    rho_1 = density_csh
    rho_2 = density_water
    rho_s = density_solvent    
    
    A = (layer_number_mean*np.pi*np.power(cylinder_radius, 2)*
         ((rho_1-rho_s)*L1+(rho_2-rho_s)*L2))
    
    P_Q = list(average_polydisperse_PQ(Q, cylinder_radius, water_height, csh_height,
                                  density_water, density_csh, density_solvent,
                                  layer_number_mean, layer_number_width))
    
    S_F = interglobuleSF(Q, fractal_dimension, fractal_cutoff, globule_radius)
    
    plt.clf()
    plt.loglog(Q, P_Q[0])
    plt.loglog(Q, S_F)
    plt.loglog(Q, P_Q[0]*S_F)
    plt.show()
    plt.clf()
    
    I_Q = (number_density* np.power(A,2) * P_Q[0] * S_F) + background
    I_err = (number_density* np.power(A,2) * P_Q[1] * S_F) + background
                                                                
    return I_Q, I_err

def main():
    
    cylinder_radius = 96.3
    water_height = 7.86
    csh_height = 3.47
    density_water = 9.469
    background = 0
    density_csh = density_water/0.043
    density_solvent = 0
    layer_number_mean = 10.86
    layer_number_width_sigma = 10.2
    layer_number_width_Z = np.power((layer_number_mean/layer_number_width_sigma),2)-1
    fractal_dimension = 2.58
    fractal_cutoff = 670
    globule_radius = 95
    number_density = 1e-20
    
    parameters = {}
    #parameters['30%']=[cylinder_radius, water_height, csh_height, density_water,
    #              background, density_csh, density_solvent, layer_number_mean,
    #              layer_number_width_Z, fractal_dimension, fractal_cutoff,
    #              globule_radius, number_density]
    
    water_height = 5.51
    fractal_dimension = 2.75
    layer_number_mean = 4.53
    layer_number_width_sigma = 2.2
    layer_number_width_Z = np.power((layer_number_mean/layer_number_width_sigma),2)-1
    globule_radius = 65.7
    density_csh = density_water/0.017
    
    parameters['10%']=[cylinder_radius, water_height, csh_height, density_water,
                  background, density_csh, density_solvent, layer_number_mean,
                  layer_number_width_Z, fractal_dimension, fractal_cutoff,
                  globule_radius, number_density]
    
    water_height = 5.82
    fractal_dimension = 2.69
    layer_number_mean = 4.73
    layer_number_width_sigma = 2.1
    layer_number_width_Z = np.power((layer_number_mean/layer_number_width_sigma),2)-1
    globule_radius = 67.4
    
    #parameters['17%']=[cylinder_radius, water_height, csh_height, density_water,
    #              background, density_csh, density_solvent, layer_number_mean,
    #              layer_number_width_Z, fractal_dimension, fractal_cutoff,
    #              globule_radius, number_density]
    
    Q = np.logspace(np.log10(1e-2),np.log10(1), 30)
    
    for humidity in parameters.keys():
        print(humidity)
        f_S = intensity(Q, *parameters[humidity])
        
        with open('{}.dat'.format(humidity), 'w') as outfile:
            outfile.write('#Humidity: {}, Model from Paper\n'.format(humidity))
            outfile.write('#Model see dx.doi.org/10.1021/jp300745g\n')
            outfile.write('#Q [Angström^-1] \t Intensity [arb. u.]\n')
            
            for q,i in zip(Q,f_S):
                outfile.write('{}\t{}\n'.format(q,i))
                
        fig = plt.figure()
        ax = plt.axes()
    
        ax.set_xlabel('Q [Angström^-1]')
        ax.set_ylabel('Intensity [arb. u.]')
        ax.set_title('Humidity: {}, Model from Paper'.format(humidity))
    
        ax.errorbar(Q,f_S[0],yerr=f_S[1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()
        plt.savefig("{}_humidity.png".format(humidity))
        plt.clf()
    
def test():
    
    cylinder_radius = 96.3
    water_height = 7.86
    csh_height = 3.47
    density_water = 9.469
    background = 0
    density_csh = density_water/0.043
    density_solvent = 0
    layer_number_mean = 10.86
    layer_number_width_sigma = 10.2
    layer_number_width_Z = np.power((layer_number_mean/layer_number_width_sigma),2)-1
    fractal_dimension = 2.58
    fractal_cutoff = 670
    globule_radius = 95
    number_density = 1e-20
    water_height = 5.51
    fractal_dimension = 2.75
    layer_number_mean = 4.53
    layer_number_width_sigma = 2.2
    layer_number_width_Z = np.power((layer_number_mean/layer_number_width_sigma),2)-1
    globule_radius = 65.7
    density_csh = density_water/0.017
    
    Q = np.logspace(np.log10(0.15),np.log10(1), 10)
    
    mus = np.linspace(0,1000,1001)/1000
    
    As = []
    Bs = []
    Cs = []
    Ds = []
    Is = []
    
    for mu in mus:
        A, B, C, D = particleSF_parts(Q, mu, cylinder_radius, water_height,
                                  csh_height, density_water, density_csh,
                                  density_solvent, layer_number_mean)
        I = np.power(D,2)*np.power(C,2)*(np.power(A,2)+np.power(B,2))
        As.append(np.power(A,2))
        Bs.append(np.power(B,2))
        Cs.append(np.power(C,2))
        Ds.append(np.power(D,2))
        Is.append(I)
    
    As = np.transpose(As)
    Bs = np.transpose(Bs)
    Cs = np.transpose(Cs)
    Ds = np.transpose(Ds)
    Is = np.transpose(Is)
    
    As = np.log10(As)
    Bs = np.log10(Bs)
    Cs = np.log10(Cs)
    Ds = np.log10(Ds)
    Is = np.log10(Is)
    
    for num,q in enumerate(Q):
        plt.plot(mus, As[num], label="A")
        plt.plot(mus, Bs[num], label="B")
        plt.plot(mus, Cs[num], label="C")
        plt.plot(mus, Ds[num], label="D")
        plt.legend()
        plt.title("{}".format(q))
        plt.show()
        plt.plot(mus, Is[num], label="I")
        plt.show()
        #plt.title('Mu: {}'.format(mu))
        #plt.loglog(Q, np.power(A,2), label="A")
        #plt.loglog(Q, np.power(B,2), label="B")
        #plt.loglog(Q, np.power(C,2), label="C")
        #plt.loglog(Q, np.power(D,2), label="D")
        #plt.legend()
        #plt.show()
        #plt.loglog(Q, np.power(D,2)*np.power(C,2)*(np.power(A,2)+np.power(B,2)))
        #plt.show()
    
    
if __name__ == '__main__':
    
    test()