
function G = Cylindergrid(r_max , r_min ,step , z_max, dens)

    % Make a cylindrical triangular grid
    arguments

        r_max(1,1) double  = log(0.22*micro*meter);
        r_min(1,1) double  = log(0.1*nano*meter);
        step(1,1) double   = 0.25;
        z_max(1,10) double = convertFrom(ones(10,1)*(15/10), nano*meter);
        dens(1,1) double   = 20;
    end
    P = [];
    for r = exp(r_min:step:r_max)
       [x,y,~] = cylinder(r ,dens);

       P = [P [x(1,:); y(1,:)]];
    end

    P = unique(P', "rows");

    G = makeLayeredGrid(triangleGrid(P), z_max);
end