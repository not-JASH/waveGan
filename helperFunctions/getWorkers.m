function getWorkers(noWorkers)
    if isempty(gcp('nocreate'))
        parpool(noWorkers);
        return
    else
        pool = gcp;
        if pool.NumWorkers ~= noWorkers
            delete(gcp);
            parpool(noWorkers)
            return
        end
    end
end