function Xprime = decoderOV(p,A,theta)

    ANorm = normalizeAdjacency(A);
    
    Z1 = theta;
    Z2 = ANorm * Z1 * p.mult1.Weights;
    Z2 = leakyrelu(Z2);
    Z2 = Z2./(1+exp(-10.*Z2));
    Z3 = ANorm * Z2 * p.mult2.Weights;
    Z3 = leakyrelu(Z3) + Z2;  
    Z3 = Z3./(1+exp(-10.*Z3));
    Z4 = ANorm * Z3 * p.mult3.Weights;
    
    Xprime = 1./(1+exp(-10.*Z4));
    

end
