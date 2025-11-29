%% preparing the initial training data  
clear; clc; close all;

num_runs = 10;
IGD_all  = zeros(num_runs,1);
TIME_all = zeros(num_runs,1);

for runid =1: 10 

    fprintf("\n=========== RUN %d ===========\n", runid);
    tic;
    d = 5;
    m = 11*d - 1;
    S_full = 2*lhsdesign(m,d,'Criterion','maximin','Iterations',100)+1;

    y_full = zeros(m,3);

    for i = 1:m
        y_full(i,:) = crash_functions(S_full(i,:));
    end

    y_c_full = 5 - y_full(:,3);   % constraint g(x) >= 0
    y_full   = y_full(:,1:2);     % only objectives

    % visualize initial points
    % figure;
    % plot(y_full(:,1), y_full(:,2), 'ob')
    % drawnow

    num_iter = 200 - m;

    %% Begin iterated exploration-exploitation loop    
    for iter = 1:num_iter
        
        % feasible mask (for objective)
        valid_idx = (y_c_full >= 0);
        S_feas = S_full(valid_idx,:);
        y_feas = y_full(valid_idx,:);

        has_feasible = ~isempty(S_feas);

        % Train constraint GP on ALL points
        GPmodel_c = fitrgp(S_full, y_c_full);

        if has_feasible
            
            if iter <= floor(2*(num_iter/3))
                % ----- Phase 1: Random weights (exploration)
                w = rand();                           % uniform weight in [0,1]
            else
                % ----- Phase 2: Adaptive weights based on PARETO points only
                [~, y_PF_feas] = nondomination(S_feas, y_feas);
                w = choose_adaptive_weight(y_PF_feas);  
            end
            % normalize only on feasible objectives
            f1 = (y_feas(:,1) - min(y_feas(:,1))) / (max(y_feas(:,1)) - min(y_feas(:,1)) + eps);
            f2 = (y_feas(:,2) - min(y_feas(:,2))) / (max(y_feas(:,2)) - min(y_feas(:,2)) + eps);

            y_train_feas = w*f1 + (1-w)*f2;

            % best feasible scalarized value
            [bestvalue, ~] = min(y_train_feas);

            % Train objective GP ONLY on feasible data
            GPmodel = fitrgp(S_feas, y_train_feas);

            % acquisition uses both GPmodel and GPmodel_c
            obj_fun = @(x) objective(x, GPmodel, GPmodel_c, bestvalue);

        else
            % no feasible yet: ignore objective GP, maximize P(feasible)
            bestvalue = inf;
            GPmodel_dummy = [];

            obj_fun = @(x) objective(x, GPmodel_dummy, GPmodel_c, bestvalue);
        end

        % Optimize acquisition
        x0 = 2*rand(1,d) + 1;
        lb = ones(1,d);
        ub = 3*ones(1,d);
        x_new = simulannealbnd(obj_fun, x0, lb, ub);

        % evaluate new point
        y_new = crash_functions(x_new);
        y_new_c = 5 - y_new(3);      % constraint
        y_new   = y_new(1,1:2);      % objectives

        % append to full data
        S_full  = [S_full;  x_new];
        y_full  = [y_full;  y_new];
        y_c_full = [y_c_full; y_new_c];

        disp(iter);
    end
        
    load Crash_PF.mat

    %% Pareto and IGD on FEASIBLE points only
    valid_idx = (y_c_full >= 0);
    S_valid = S_full(valid_idx,:);
    y_valid = y_full(valid_idx,:);

    [~, y_Pareto] = nondomination(S_valid, y_valid);
    
    figure; hold on;

    % Plot VALID (feasible) points
    plot(y_valid(:,1), y_valid(:,2), 'ob', 'MarkerFaceColor','b');  % feasible points in blue
    
    % Plot INVALID points
    invalid_idx = (y_c_full < 0);
    y_invalid = y_full(invalid_idx,:);
    plot(y_invalid(:,1), y_invalid(:,2), 'o', ...
        'MarkerFaceColor',[0.6 0.6 0.6], ...   % gray fill
        'MarkerEdgeColor',[0.6 0.6 0.6], ...   % gray edge
        'MarkerSize', 6);                      % same size as blue
   
    % Plot Pareto points
    plot(y_Pareto(:,1), y_Pareto(:,2), 'or', 'MarkerFaceColor','r');
    
    % Plot reference PF
    plot(Crash_PF(:,1), Crash_PF(:,2), 'xk');
    
    title(sprintf('Run %d Pareto Front', runid));
    legend('Feasible points','Infeasible points','Pareto Front','Reference PF');

    saveas(gcf, sprintf('Run_%d_plot.fig', runid));

    IGD_all(runid) = inverted_generational_distance(y_Pareto, Crash_PF);
    TIME_all(runid) = toc;

    fprintf("Run %d IGD = %.5f  Time = %.4f sec\n", runid, IGD_all(runid), TIME_all(runid));

end 

%% Save results
save IGD_results.mat IGD_all;
save Time_results.mat TIME_all;

fprintf("\n==== All runs completed ====\n");


%% preparing the initial training data  
clear; clc; close all;

num_runs = 10;
IGD_all  = zeros(num_runs,1);
TIME_all = zeros(num_runs,1);

for runid = 1:num_runs
    
    fprintf("\n=========== RUN %d ===========\n", runid);
    rng(runid);                    % <--- reproducible random seed
    tic;

    d = 5;
    m = 11*d - 1;

    S_full = 2*lhsdesign(m,d,'Criterion','maximin','Iterations',100)+1;

    y_full = zeros(m,3);
    for i = 1:m
        y_full(i,:) = crash_functions(S_full(i,:));
    end

    y_c_full = 5 - y_full(:,3);   % constraint g(x) >= 0
    y_full   = y_full(:,1:2);     % keep only f1, f2

    num_iter = 200 - m;

    %% Iterated loop
    for iter = 1:num_iter
        
        valid_idx = (y_c_full >= 0);
        S_feas    = S_full(valid_idx,:);
        y_feas    = y_full(valid_idx,:);
        has_feasible = ~isempty(S_feas);

        % Train constraint GP on ALL points
        GPmodel_c = fitrgp(S_full, y_c_full);

        if has_feasible
            % adaptive weight using feasible objectives
            if iter <= floor(2*(num_iter/3))
                % Random weights (exploration)
                w = rand();                      
            else
                %Adaptive weights based on PARETO points only
                [~, y_PF_feas] = nondomination(S_feas, y_feas);
                w = choose_adaptive_weight(y_PF_feas);  
            end
            % Normalize feasible objectives
            f1 = (y_feas(:,1) - min(y_feas(:,1))) ...
                ./ (max(y_feas(:,1)) - min(y_feas(:,1)) + eps);
            f2 = (y_feas(:,2) - min(y_feas(:,2))) ...
                ./ (max(y_feas(:,2)) - min(y_feas(:,2)) + eps);

            y_train_feas = w*f1 + (1-w)*f2;

            bestvalue = min(y_train_feas);

            % Train objective surrogate *only on feasible pts*
            GPmodel = fitrgp(S_feas, y_train_feas);

            obj_fun = @(x) objective(x,GPmodel,GPmodel_c,bestvalue);

        else
            %% No feasible yet — ignore objective model
            GPmodel = [];
            bestvalue = inf;

            obj_fun = @(x) objective(x,GPmodel,GPmodel_c,bestvalue);
        end

        %% Optimize acquisition using SA
        x0 = 2*rand(1,d) + 1;
        lb = ones(1,d);
        ub = 3*ones(1,d);
        x_new = simulannealbnd(obj_fun, x0, lb, ub);

        % Evaluate new point
        y_new = crash_functions(x_new);
        y_c_new = 5 - y_new(3);
        y_new = y_new(1:2);

        % Append
        S_full  = [S_full; x_new];
        y_full  = [y_full; y_new];
        y_c_full = [y_c_full; y_c_new];

        disp(iter);
    end

    load Crash_PF.mat

    %% Compute Pareto front on feasible only
    valid_idx = (y_c_full >= 0);
    S_valid = S_full(valid_idx,:);
    y_valid = y_full(valid_idx,:);

    [~, y_Pareto] = nondomination(S_valid, y_valid);

    %% Plot
    figure; hold on;
    plot(y_valid(:,1), y_valid(:,2), 'ob', 'MarkerFaceColor','b');
    
    invalid_idx = (y_c_full < 0);
    y_invalid = y_full(invalid_idx,:);
    plot(y_invalid(:,1), y_invalid(:,2), 'o', ...
        'MarkerFaceColor',[0.6 0.6 0.6], ...
        'MarkerEdgeColor',[0.6 0.6 0.6], ...
        'MarkerSize', 6);

    plot(y_Pareto(:,1), y_Pareto(:,2), 'or', 'MarkerFaceColor','r');
    plot(Crash_PF(:,1), Crash_PF(:,2), 'xk');

    title(sprintf('Run %d Pareto Front', runid));
    legend('Feasible','Infeasible','Obtained PF','Reference PF');

    saveas(gcf, sprintf('Run_%d_plot.fig', runid));

    IGD_all(runid)  = inverted_generational_distance(y_Pareto, Crash_PF);
    TIME_all(runid) = toc;

    fprintf("Run %d IGD = %.5f  Time = %.4f sec\n", ...
            runid, IGD_all(runid), TIME_all(runid));

end

save IGD_results.mat IGD_all;
save Time_results.mat TIME_all;

fprintf("\n==== All runs completed ====\n");


%% Objective (Acquisition) Function
function f_p = objective(x, GPmodel, GPmodel_c, bestvalue)

    P = 1e6;        % penalty

    % Predict constraint
    g = predict(GPmodel_c, x);

    % If no feasible discovered yet → minimize violation
    if isinf(bestvalue)
        f_p = P * max(0, -g);    % penalize only
        return;
    end

    % Predict scalarized objective
    [mu, sigma] = predict(GPmodel, x);
    sigma = max(sigma,1e-9);

    % Expected Improvement
    z = (bestvalue - mu) ./ sigma;
    EI = (bestvalue - mu).*normcdf(z) + sigma.*normpdf(z);

    % Penalized objective
    f_p = -EI + P * max(0, -g);
end


