function dlx = gpudl(x,labels)
    dlx = gpuArray(dlarray(x,labels));
end