function parameter = initGauss(parameterSize,sigma)
    if nargin < 2
        sigma = 0.05;
    end
    % single -> double 
    parameter = randn(parameterSize,'double') .* sigma;
end