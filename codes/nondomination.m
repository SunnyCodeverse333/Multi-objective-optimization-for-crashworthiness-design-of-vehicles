
function [x_P, y_P] = nondomination(S,y)
    
    m = size(y,1);
    nondominated = true(m,1);
    for i = 1:m % Loop through the dataset of evaluated solutions
        f1=y(i,1);
        f2 = y(i,2);
        for j = 1:m
            if f1 >= y(j,1) && f2 >= y(j,2) % Check if ith solution is dominated by the jth solution
                if f1 > y(j,1) || f2 > y(j,2)
                    nondominated(i) = false;
                    break % If dominated, break out of inner for loop and move to the next solution in the outer loop
                end
            end
        end
    end
    x_P = S(nondominated,:);
    y_P = y(nondominated,:);
end

