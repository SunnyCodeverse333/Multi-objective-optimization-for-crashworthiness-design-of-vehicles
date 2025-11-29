function IGD = inverted_generational_distance(y, Pareto_front)
    
    m_P = size(Pareto_front,1);

    % Normalize obtained Pareto front approximation
    y(:,1) = (y(:,1) - min(Pareto_front(:,1)))/(max(Pareto_front(:,1))-min(Pareto_front(:,1)));
    y(:,2) = (y(:,2) - min(Pareto_front(:,2)))/(max(Pareto_front(:,2))-min(Pareto_front(:,2)));

    % Normalize the true Pareto front
    Pareto_front(:,1) = (Pareto_front(:,1) - min(Pareto_front(:,1)))/(max(Pareto_front(:,1))-min(Pareto_front(:,1)));
    Pareto_front(:,2) = (Pareto_front(:,2) - min(Pareto_front(:,2)))/(max(Pareto_front(:,2))-min(Pareto_front(:,2)));

    IGD = 0;
    for i = 1:m_P
        y_diff = [y(:,1)-Pareto_front(i,1), y(:,2)-Pareto_front(i,2)];
        y_norm = sqrt(y_diff(:,1).^2 + y_diff(:,2).^2);
        IGD = IGD + min(y_norm);
    end

    IGD = IGD/m_P;
end

