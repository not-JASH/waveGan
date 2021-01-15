function [dly,st] = batchnormwrap(dlx,params,st,i)
    if isempty(st.BN{i})
        [dly,st.BN{i}.mu,st.BN{i}.sig] = batchnorm(dlx,params.BNo{i},params.BNs{i},"MeanDecay",0.8);
    else
        [dly,st.BN{i}.mu,st.BN{i}.sig] = batchnorm(dlx,params.BNo{i},params.BNs{i},...
            st.BN{i}.mu,st.BN{i}.sig,"MeanDecay",0.8);
    end
end