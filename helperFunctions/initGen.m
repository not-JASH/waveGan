function [Gen,stGen] = initGen(genFilt)
    if rem(genFilt,2)~=0;genFilt=genFilt+1;end
    Gen.CNw = []; Gen.CNb = [];
    Gen.BNo = []; Gen.BNs = [];
    Gen.TCw = []; Gen.TCb = [];
    stGen.BN = [];
    
    filterSize = [4 4];
    chpGr = 2;
    fpGr = genFilt/4;
    nGr = [7 4 4 4 4 2 1];
    i = 1;
    fprintf("convolution sizes\n")
    addConv;
    for i = 2:7
        chpGr = fpGr*nGr(i-1)/nGr(i); 
        fpGr = 2*fpGr;
        addConv;
        addBatchNorm(fpGr*nGr(i));
        stGen.BN = [stGen.BN;{[]}];
    end
    nGr = flip(nGr);i = 1;
    fprintf("transposedCovolution sizes\n")
    addTconv;addBatchNorm(fpGr*nGr(i));stGen.BN = [stGen.BN;{[]}];
    for i = 2:6
        chpGr = (fpGr*nGr(i-1))/nGr(i) + size(Gen.CNw{7-i+1},4);
        fpGr = 0.5*fpGr;
        addTconv;
        addBatchNorm(fpGr*nGr(i));
        stGen.BN = [stGen.BN;{[]}];
    end
    fpGr = 7;chpGr = 60;nGr(7) = 2;i = 7;
    addTconv;

    function addBatchNorm(batchNormSize)
        Gen.BNo = [Gen.BNo;{dlarray(zeros(batchNormSize,1,'single'))}];
        Gen.BNs = [Gen.BNs;{dlarray(ones(batchNormSize,1,'single'))}];
    end
  
    function addConv
        fprintf("%d\t%d\t%d\t%d\t%d\n",[filterSize,chpGr,fpGr,nGr(i)]);
        Gen.CNw = [Gen.CNw;{...
            dlarray(initGauss([filterSize,chpGr,fpGr,nGr(i)]),'SSCUU')}];
        Gen.CNb = [Gen.CNb;{...
            dlarray(zeros(fpGr*nGr(i),1),'C')}];
    end
    
    function addTconv
        fprintf("%d\t%d\t%d\t%d\t%d\n",[filterSize,fpGr,chpGr,nGr(i)]);
        Gen.TCw = [Gen.TCw;{...
            dlarray(initGauss([filterSize,fpGr,chpGr,nGr(i)]),'SSCUU')}];
        Gen.TCb = [Gen.TCb;{...
            dlarray(zeros(fpGr*nGr(i),1),'C')}];    
    end
end