function [LA,x1,y1,x2,y2] = getLA(Xul,Yul,Xll,Yll)
% Compute lip aperture.
    [D,x1,y1,x2,y2] = spanPolyPolyDist_modified(Xul,Yul,Xll,Yll);
    LA = min(D);
    x1=x1(1); x2=x2(1); y1=y1(1); y2=y2(1);
end