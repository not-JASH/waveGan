function [H,edges] = hEntropy(data,n)
  
    [f,edges] = histcounts(data,n);
    w = edges(2)-edges(1);
    H = f.*log(f/w);
    H(isnan(H))=[];
    H = sum(H);
end