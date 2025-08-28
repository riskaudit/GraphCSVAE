function B = sparse_dilate_ultra(A)
    % A: sparse binary matrix
    % B: 8-connected dilation of A

    [m, n] = size(A);
    [r, c] = find(A);
    numPts = numel(r);

    % Preallocate for max possible indices (9 shifts)
    maxIdx = numPts * 9;
    rows = zeros(maxIdx, 1);
    cols = zeros(maxIdx, 1);
    idx = 0;

    % Define shifts manually (8-connected + center)
    shifts = [-1, -1; -1, 0; -1, 1;
               0, -1;  0, 0;  0, 1;
               1, -1;  1, 0;  1, 1];

    for s = 1:9
        dr = shifts(s,1);
        dc = shifts(s,2);

        rr = r + dr;
        cc = c + dc;

        % Filter in-bounds
        mask = (rr >= 1 & rr <= m & cc >= 1 & cc <= n);
        len = nnz(mask);

        if len > 0
            rows(idx+1:idx+len) = rr(mask);
            cols(idx+1:idx+len) = cc(mask);
            idx = idx + len;
        end
    end

    % Trim and build sparse binary matrix
    rows = rows(1:idx);
    cols = cols(1:idx);
    B = sparse(rows, cols, 1, m, n);
    B = B > 0;
end
