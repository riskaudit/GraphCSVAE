function [theta, theta_sampled] = encoderOV(p,X,A,mask)

    ANorm = normalizeAdjacency(A);

    % temp = 10; % for Case Study 1. Quezon City
    temp = 1; % for Case Study 2 and 3. Khurushkul (BGD), Freetown (SLE), and METEOR 2.5D
    
    Z1 = X;
    Z2 = ANorm * Z1 * p.mult1.Weights;
    Z2 = tanh(Z2);
    Z3 = ANorm * Z2 * p.mult2.Weights; 
    Z3 = tanh(Z3) + Z2;
    Z4 = ANorm * Z3 * p.mult3.Weights; 

    logit = Z4.*(mask>0) + (-1e3).*(mask==0);
    gk = -log(-log(rand(size(mask))));
    theta_sampled = exp((logit+gk)./temp)./sum(exp((logit+gk)./temp),2);
    theta = exp(logit)./sum(exp(logit),2);

end