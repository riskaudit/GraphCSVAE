function [fig31,fig32,lineReconLoss,lineKL] = initializeOVfig()

    C = colororder;

    fig31 = figure(31); clf
    lineReconLoss.Train = animatedline(Color=C(2,:));
    lineReconLoss.Valid = animatedline(LineStyle="--", Marker="o", MarkerFaceColor="r");
    lineReconLoss.Test = animatedline(LineStyle="--", Marker="o", MarkerFaceColor="b");
    ylim([-inf inf]); xlabel("Iteration"); ylabel("ReconLoss"); yscale linear; grid on
    
    fig32 = figure(32); clf
    lineKL.Train = animatedline(Color=C(2,:));
    lineKL.Valid = animatedline(LineStyle="--", Marker="o", MarkerFaceColor="r");
    lineKL.Test = animatedline(LineStyle="--", Marker="o", MarkerFaceColor="b");
    ylim([-inf inf]); xlabel("Iteration"); ylabel("KL"); yscale linear; grid on
    
end

