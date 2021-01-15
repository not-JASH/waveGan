function [Dis,stDis] = initDis(discFilt)
%     coeffs = [1,2,4;2,4,8];
    Dis.CNw = [];Dis.CNb = [];
    Dis.BNo = [];Dis.BNs = [];
    stDis.BN = [];

    filterSize = [4 4];
    chpGr = 2;
    fpGr = 4*discFilt;
    nGr = [7 4 4 4 1];
    i = 1;
    addConv;

    for i = 2:4
        chpGr = fpGr*nGr(i-1)/nGr(i);
        fpGr = fpGr/2;
        addConv;
        addBatchNorm(fpGr*nGr(i));
        stDis.BN = [stDis.BN;{[]}];
    end
    i = 5;
    chpGr = fpGr*nGr(i-1)/nGr(i);
    fpGr = 1;
    addConv;


    function addBatchNorm(batchNormSize)
        Dis.BNo = [Dis.BNo;{dlarray(zeros(batchNormSize,1,'single'))}];
        Dis.BNs = [Dis.BNs;{dlarray(ones(batchNormSize,1,'single'))}];
    end

    function addConv
%         fprintf("%d\t%d\t%d\t%d\t%d\n",[filterSize,chpGr,fpGr,nGr(i)]);
        Dis.CNw = [Dis.CNw;{...
            dlarray(initGauss([filterSize,chpGr,fpGr,nGr(i)]),'SSCUU')}];
        Dis.CNb = [Dis.CNb;{...
            dlarray(zeros(fpGr*nGr(i),1),'C')}];
    end
end