%% preparing the initial training data  
clear; clc; close all;

num_runs = 10;
IGD_all  = zeros(num_runs,1);
TIME_all = zeros(num_runs,1);

for runid =1: num_runs 

    fprintf("\n=========== RUN %d ===========\n", runid);
    tic;
    d = 5;
    m = 11*d - 1;

    % Initial sample points
    S_full = 2*lhsdesign(m,d,'Criterion','maximin','Iterations',100)+1;

    y_full = zeros(m,3);
    for i = 1:m
        y_full(i,:) = crash_functions(S_full(i,:));
    end

    y_c_full = 5 - y_full(:,3);  % constraint g(x) >= 0
    y_full   = y_full(:,1:2);    % objectives only

    num_iter = 200 - m;
    lamda = 1e-6;

    for iter = 1:num_iter
        % Feasible points
        valid_idx = y_c_full >= 0;
        S_feas = S_full(valid_idx,:);
        y_feas = y_full(valid_idx,:);
        has_feasible = ~isempty(S_feas);

        if has_feasible
   
            if iter <= floor(2*(num_iter/3))
                w = rand();                          
            else
                % Adaptive weights based on PARETO points only
                [~, y_PF_feas] = nondomination(S_feas, y_feas);
                w = choose_adaptive_weight(y_PF_feas);  
            end
            f1 = (y_feas(:,1) - min(y_feas(:,1))) / (max(y_feas(:,1)) - min(y_feas(:,1)) + eps);
            f2 = (y_feas(:,2) - min(y_feas(:,2))) / (max(y_feas(:,2)) - min(y_feas(:,2)) + eps);
            y_train = w*f1 + (1-w)*f2;


            bestvalue = min(y_train);
        else
            y_train = [];
            bestvalue = inf;
        end

        % --- Ensemble Surrogate Models ---
        len_scale = mean(pdist(S_feas));
        len_scale_c = mean(pdist(S_full));
        % 1) RBF
        D = pdist2(S_feas, S_feas);
        K = exp(-(D.^2)/(2*len_scale^2));
        w_opt = pinv(K + lamda*eye(size(K,1))) * y_train;

        % 2) Decision Tree
        DTmodel = fitrtree(S_feas, y_train);

        % 3) Gaussian Process
        if ~isempty(y_train)
            GPmodel = fitrgp(S_feas, y_train);
        else
            GPmodel = [];
        end

        % --- Constraint Surrogate Models ---
        % RBF
        D = pdist2(S_full, S_full);
        K_c = exp(-(D.^2)/(2*len_scale_c^2));
        w_opt_c = pinv(K_c + lamda*eye(size(K_c,1))) * y_c_full;
        % Decision Tree
        DTmodel_c = fitrtree(S_full, y_c_full);
        % GP
        GPmodel_c = fitrgp(S_full, y_c_full);

        % --- Acquisition Function ---
        obj_fun = @(x) objective(x, w_opt, DTmodel, GPmodel, ...
                                 w_opt_c, DTmodel_c, GPmodel_c, S_full,S_feas, len_scale , len_scale_c, bestvalue);

        % Optimize acquisition (simulated annealing)
        x0 = 2*rand(1,d) + 1;
        lb = ones(1,d);
        ub = 3*ones(1,d);
        x_new = simulannealbnd(obj_fun, x0, lb, ub);

        % Evaluate true function
        y_new_full = crash_functions(x_new);
        y_c_new = 5 - y_new_full(3);
        y_new = y_new_full(1:2);

        % Append new sample
        S_full = [S_full; x_new];
        y_full = [y_full; y_new];
        y_c_full = [y_c_full; y_c_new];

        fprintf('iter %d done\n', iter);
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


function f_p = objective(x, w_opt, DTmodel, GPmodel, w_opt_c, DTmodel_c, GPmodel_c, S_full ,S_feas, len_scale , len_scale_c, bestvalue)
    %% ---- Predict objective ----
    k_vec = exp(-sum((S_feas - x).^2, 2)/(2*len_scale^2));
    mu_rbf = k_vec' * w_opt;
    mu_dt  = predict(DTmodel, x);
    mu_gp  = predict(GPmodel, x);
    mu = mean([mu_rbf, mu_dt, mu_gp]);

    sigma = std([mu_rbf, mu_dt, mu_gp], 1);
    sigma = max(sigma, 1e-8);

    %% ---- Predict constraint ----
    k_vec_c = exp(-sum((S_full - x).^2,2)/(2*len_scale_c^2));
    mu_rbf_c = k_vec_c' * w_opt_c;
    mu_dt_c  = predict(DTmodel_c, x);
    mu_gp_c  = predict(GPmodel_c, x);
    mu_c = mean([mu_rbf_c, mu_dt_c, mu_gp_c]);

    sigma_c = std([mu_rbf_c, mu_dt_c, mu_gp_c], 1);
    sigma_c = max(sigma_c, 1e-8);

    %% ---- Constrained EI ----
    if bestvalue == inf
        f_p = -normcdf(mu_c / sigma_c);  % only feasibility search
        return;
    end

    z = (bestvalue - mu) / sigma;
    EI = (bestvalue - mu) * normcdf(z) + sigma * normpdf(z);
    Pf = normcdf(mu_c / sigma_c);

    f_p = -(EI * Pf);
end





