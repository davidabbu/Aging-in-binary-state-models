# FUNCTIONS

function binomial_dist(k,m,q)
    return binomial(k,m)*(q^m)*(1-q)^(k-m)
end

function rho(k_min,k_max,z,infected)
    suma = 0
    for k in k_min:k_max
        suma = suma + dist(z,k)*sum(infected[k+1,:,:])
    end
    return suma
end

function interface(k_min,k_max,z,susceptible)
    suma = 0
    for k in k_min:k_max
        suma_m = 0
        for m in 0:k
            suma_m = suma_m + m*sum(susceptible[k+1,m+1,:])
        end
        suma = suma + dist(z,k)*suma_m
    end
    return suma
end

function coef_s(k_min,k_max,z,F,susceptible)
    num = 0
    denom = 0
    for k in k_min:k_max
        suma = 0
        suma2 = 0
        for m in 0:k
            suma = suma + (k-m)*sum(F[k+1,m+1,:].*susceptible[k+1,m+1,:])
            suma2 = suma2 + (k-m)*sum(susceptible[k+1,m+1,:])
        end
        num = num + dist(z,k)*suma
        denom = denom + dist(z,k)*suma2
    end
    denom = denom + 10^(-12)
    return num/denom
end

function coef_i(k_min,k_max,z,F,susceptible)
    num = 0
    denom = 0
    for k in k_min:k_max
        suma = 0
        suma2 = 0
        for m in 0:k
            suma = suma + m*sum(F[k+1,m+1,:].*susceptible[k+1,m+1,:])
            suma2 = suma2 + m*sum(susceptible[k+1,m+1,:])
        end
        num = num + dist(z,k)*suma
        denom = denom + dist(z,k)*suma2
    end
    denom = denom + 10^(-12)
    return num/denom
end

# FUNCTION FOR ODE
function AME(du,u,p,t)
    F,R,FE,RE,FA,RA = p
    
    beta_s = coef_s(k_min,k_max,z,F,u[1,:,:,:])
    beta_i = coef_i(k_min,k_max,z,F,u[1,:,:,:])
    gamma_s = coef_s(k_min,k_max,z,R,u[2,:,:,:])
    gamma_i = coef_i(k_min,k_max,z,R,u[2,:,:,:])
    
    # TERMS j = 0 for all m
    for k in k_min:k_max
        for m in 0:k
            
            # Susceptible
            beta_term = - (k-m)*beta_s*u[1,k+1,m+1,tau_min+1]
            gamma_term = - m*gamma_s*u[1,k+1,m+1,tau_min+1]
            du[1,k+1,m+1,tau_min+1] = - u[1,k+1,m+1,tau_min+1] + sum(R[k+1,m+1,1:end].*u[2,k+1,m+1,1:end]) + sum(FR[k+1,m+1,1:end].*u[1,k+1,m+1,1:end]) + beta_term + gamma_term
            
            # Infected
            beta_term = - (k-m)*beta_i*u[2,k+1,m+1,tau_min+1]
            gamma_term = - m*gamma_i*u[2,k+1,m+1,tau_min+1]
            du[2,k+1,m+1,tau_min+1] = - u[2,k+1,m+1,tau_min+1] + sum(F[k+1,m+1,1:end].*u[1,k+1,m+1,1:end]) + sum(RR[k+1,m+1,1:end].*u[2,k+1,m+1,1:end]) + beta_term + gamma_term
        end
    end
    
    # TERMS with j > 0
    for k in k_min:k_max
        for m in 0:k
            for j in tau_min+1:tau_max
                
                # Susceptible                
                if m == 0
                    terms = - (k - m)*beta_s*u[1,k+1,m+1,j+1] - m*gamma_s*u[1,k+1,m+1,j+1] + (m + 1)*gamma_s*u[1,k+1,m+2,j]
                elseif m == k_max
                    terms = (k - m + 1)*beta_s*u[1,k+1,m,j] - (k - m)*beta_s*u[1,k+1,m+1,j+1] - m*gamma_s*u[1,k+1,m+1,j+1]
                else
                    terms = (k - m + 1)*beta_s*u[1,k+1,m,j] - (k - m)*beta_s*u[1,k+1,m+1,j+1] - m*gamma_s*u[1,k+1,m+1,j+1] + (m+1)*gamma_s*u[1,k+1,m+2,j]
                end
                du[1,k+1,m+1,j+1] = -u[1,k+1,m+1,j+1] + FA[k+1,m+1,j]*u[1,k+1,m+1,j] + terms
                
                # Infected
                if m == 0
                    terms = - (k - m)*beta_i*u[2,k+1,m+1,j+1] - m*gamma_i*u[2,k+1,m+1,j+1] + (m + 1)*gamma_i*u[2,k+1,m+2,j]
                elseif m == k_max
                    terms = (k - m + 1)*beta_i*u[2,k+1,m,j] - (k - m)*beta_i*u[2,k+1,m+1,j+1] - m*gamma_i*u[2,k+1,m+1,j+1]
                else
                    terms = (k - m + 1)*beta_i*u[2,k+1,m,j] - (k - m)*beta_i*u[2,k+1,m+1,j+1] - m*gamma_i*u[2,k+1,m+1,j+1] + (m+1)*gamma_i*u[2,k+1,m+2,j]
                end
                du[2,k+1,m+1,j+1] = -u[2,k+1,m+1,j+1] + FA[k+1,m+1,j]*u[2,k+1,m+1,j] + terms + terms
            end
        end
    end  
end

function solve_AME(t_span,dti,ds,method,PARAMS)
    prob = ODEProblem(AME,u0,t_span,PARAMS)
    sol = solve(prob, method, dt=dti,saveat=ds)
    return sol
end

function infected_from(sol)
    arr_rho = []
    for index in 1:length(sol)
        u_s = sol[index]
        push!(arr_rho,rho(k_min,k_max,z,u_s[2,:,:,:]))
    end
    return arr_rho
end

function susceptible_from(sol)
    arr_rho = []
    for index in 1:length(sol)
        u_s = sol[index]
        push!(arr_rho,rho(k_min,k_max,z,u_s[1,:,:,:]))
    end
    return arr_rho
end

function interface_from(sol)
    arr_rho = []
    for index in 1:length(sol)
        u_s = sol[index]
        push!(arr_rho,interface(k_min,k_max,z,u_s[1,:,:,:]))
    end
    return arr_rho
end