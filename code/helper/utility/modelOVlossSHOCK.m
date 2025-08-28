function [  grad_pOV_encoder, grad_pOV_decoder, ...
            reconLoss, KL, theta_sampled, X_prime] = modelOVlossSHOCK(...
            pOV_encoder, pOV_decoder, X_TMM, A, p)

    [theta, theta_sampled] = encoderOV(pOV_encoder,X_TMM,A,p);
    X_prime = decoderOV(pOV_decoder,A,theta_sampled);
    reconLoss = mse(X_TMM, X_prime, 'DataFormat','BC');

    idx = sum(p,2)==1;
    KL1 = (theta(idx,:).*log((theta(idx,:)+(1e-6))./(p(idx,:)+(1e-6)))); 
    KL1(isnan(KL1)) = 0;
    KL1 = mean(sum(KL1,2));
    KL2 = mean(-sum(p(idx,:).*log((theta(idx,:)+(1e-6))), 2));
    disp([KL1, KL2]) 

    reconLoss = reconLoss.*1e2;
    KL = (KL1+KL2).*1e3;
    loss = reconLoss + KL;
    [grad_pOV_encoder,grad_pOV_decoder] = dlgradient(loss,pOV_encoder,pOV_decoder);
    disp([sum(isnan(extractdata(grad_pOV_encoder.mult3.Weights)), 'all'), ...
                  sum(isnan(extractdata(grad_pOV_decoder.mult3.Weights)), 'all')])


end