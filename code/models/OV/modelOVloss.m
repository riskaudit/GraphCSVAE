function [  grad_pOV_encoder, grad_pOV_decoder, ...
            reconLoss, KL, sampled_theta, X_prime] = modelOVloss(...
            pOV_encoder, pOV_decoder, X_TMM, A, p, wX, wP, w_prior, train)
    
    % Identify the most dominant class
    [~, idx] = max(p, [], 2);
    B = zeros(size(p));
    lin_idx = sub2ind(size(p), (1:size(p,1))', idx);
    B(lin_idx) = 1;
    
    % Forward operation
    [theta, sampled_theta] = encoderOV(pOV_encoder,X_TMM(:,1),A,p);
    X_prime = decoderOV(pOV_decoder,A,sampled_theta);

    % Calculate Reconstruction Loss
    reconLoss = wQReconLoss( X_TMM(:,1), X_prime, wX(:,1) );

    % Calculate KL Divergence loss
    KL1 = ((theta+1e-6).*log((theta+1e-6)./(B+1e-6))); 
    KL1(isnan(KL1)) = 0;
    KL1 = mean( (sum(KL1.*(p>0)./sum(p>0,2),2)) .* sum(B.*p.*p.*wP.*1e6.*w_prior, 2) );

    % Calculate Supervised Cross-Entropy loss
    KL2 = B.*log(theta+1e-6);
    KL2(isnan(KL2)) = 0;
    KL2 = mean(-sum(KL2, 2).* sum(B.*p.*p.*wP.*1e6.*w_prior, 2));


    % Combine and obtain the gradient
    reconLoss = reconLoss*1e10;
    KL = KL1+KL2;
    loss = reconLoss + KL;
    [grad_pOV_encoder,grad_pOV_decoder] = dlgradient(loss,pOV_encoder,pOV_decoder);
    disp([sum(isnan(extractdata(grad_pOV_encoder.mult3.Weights)), 'all'), ...
                  sum(isnan(extractdata(grad_pOV_decoder.mult3.Weights)), 'all')])


end