function w = choose_adaptive_weight(Y)
    % Y: N×2 matrix of feasible objective values 
    if size(Y,1) < 2
        w = rand(); return;
    end
    %  Normalize objectives
    Y = (Y - min(Y)) ./ (max(Y)-min(Y) + eps);

    % Compute angles theta = atan2(f2, f1), in [0, pi/2]
    theta = atan2(Y(:,2), Y(:,1));

    % Add fixed boundary angles: 0 and pi/2
    theta = [0; theta; pi/2];

    % Sort angles
    theta = sort(theta);

    % Find the largest interior gap
    [~, idx] = max(diff(theta));
    mid_theta = 0.5 * (theta(idx) + theta(idx+1));

    %  Convert mid-angle to weight
    w = cos(mid_theta) / (cos(mid_theta) + sin(mid_theta));

    % Clamp weight to [0,1]
    w = max(0, min(1, w));
end
