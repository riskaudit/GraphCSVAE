function [pOV_encoder,pOV_decoder] = initializePOV(nL,nX)

    nH = 24;
    pOV_encoder = struct;
    pOV_encoder.mult1.Weights = initializeGlorot([nX nH],nH,nX,"double");
    pOV_encoder.mult2.Weights = initializeGlorot([nH nH],nH,nH,"double");
    pOV_encoder.mult3.Weights = initializeGlorot([nH nL],nL,nH,"double");
    
    pOV_decoder = struct;
    pOV_decoder.mult1.Weights = initializeGlorot([nL nH],nH,nL,"double");
    pOV_decoder.mult2.Weights = initializeGlorot([nH nH],nH,nH,"double");
    pOV_decoder.mult3.Weights = initializeGlorot([nH nX],nX,nH,"double");

    
end

