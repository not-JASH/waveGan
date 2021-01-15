function val = estimators(x,k,type,name)
    val = x;
    switch name
        case 'L2'
            estimator = @L2;
        case 'L1'
            estimator = @L1;
        case 'L1-L2'
            estimator = @L1_L2;
        case 'Lp'
            estimator = @Lp;
        case 'fair'
            estimator = @fair;
        case 'Huber'
            estimator = @huber;
        case 'Cauchy'
            estimator = @cauchy;
        case 'Welsch' 
            estimator = @welsch;
        case 'Tukey'
            estimator = @tukey;
    end

    for i = 1:length(x)
        feval(estimator);
    end
    
    function L2
        switch type 
            case 'loss'
                val(i) = power(x(i),2)/2;
            case 'influence'
                val(i) = x(i);
            case 'weight'
                val(i) = 1;
        end
    end

    function L1
        switch type
            case 'loss'
                val(i) = abs(x(i));
            case 'influence'
                val(i) = sign(x(i));
            case 'weight'
                val(i) = 1/abs(x(i));
        end
    end

    function L1_L2
       switch type
            case 'loss'
                val(i) = 2*(sqrt(1+power(x(i),2)/2)-1);
            case 'influence'
                val(i) = x(i)/sqrt(1+power(x(i),2)/2);
            case 'weight'
                val(i) = 1/sqrt(1+power(x(i),2)/2);
       end
    end

    function Lp
        switch type
            case 'loss'
                val(i) = power(abs(x(i)),k)/k;
            case 'influence'
                val(i) = sign(x(i))*power(abs(x(i)),k-1);
            case 'weight'
                val(i) = power(abs(x(i)),k-2);
        end
    end

    function fair
        switch type
            case 'loss'
                val(i) = power(k,2)*(abs(x(i))/k - log10(1+abs(x(i))/k));
            case 'influence'
                val(i) = x(i)/(1+abs(x(i))/k);
            case 'weight'
                val(i) = 1/(1+abs(x(i))/k);
        end
    end

    function huber
        switch type
            case 'loss'
                if abs(x(i))<=k
                    val(i) = power(x(i),2)/2;
                else
                    val(i) = k*(abs(x(i))-k/2);
                end
            case 'influence'
                if abs(x(i))<=k
                    val(i) = x(i);
                else 
                    val(i) = k*sign(x(i));
                end
            case 'weight'
                if abs(x(i))<=k
                    val(i) = 1;
                else
                    val(i) = k/abs(x(i));
                end
        end
    end

    function cauchy
        switch type
            case 'loss'
                val(i) = power(k,2)*log10(1+power(x(i)/k,2))/2;
            case 'influence'
                val(i) = x(i)/(1+power(x(i)/k,2));
            case 'weight'
                val(i) = exp(-power(x(i)/k,2));
        end
    end

    function welsch
        switch type
            case 'loss'
                val(i) = power(k,2)*(1-exp(-power(x(i)/k,2)))/1;
            case 'influence'
                val(i) = x*exp(-power(x(i)/k,2));
            case 'weight'
                val(i) = exp(-power(x(i)/c,2));
        end
    end
    
    function tukey
        switch type
            case 'loss'
                if abs(x(i))<=k
                    val(i) = power(k,2)*(1-power(1-power(x(i)/k,2),3))/6;
                else
                    val(i) = power(k,2);
                end
            case 'influence'
                if abs(x(i))<=k
                    val(i) = x(i)*power(1-power(x(i)/k,2),2);
                else
                    val(i) = 0;
                end
            case 'weight'
                if abs(x(i))<=k
                    val(i) = power(1-power(x(i)/k,2),2);
                else
                    val(i) = 0;
                end
        end
    end
end