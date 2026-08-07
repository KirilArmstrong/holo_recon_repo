using TestImages, SyntheticObjects
using IndexFunArrays


function generate_object_2d(;type="Image", angle=[0,0], lam=0.45)
    N = 512;
    angle .= Float32.(angle)
    lam = Float32(lam)
    if type == "Image"
        # generate resolution sample object
        obj = Float32.(testimage("resolution512"));
        obj_height = obj./maximum(obj) # normalize to 1 um
        obj_phi = obj_height./lam .*4*pi;
        obj_amp = 1;
        phase_obj = obj_amp.*exp.(-1im.*obj_phi);
        phase_obj = reshape(phase_obj, (size(phase_obj)...,1))
    elseif type == "Plane"
        # generate a constant plane phase images
        obj = ones(N,N).* exp.(-1im.*Float32.(pi/4)); #rand(range(-pi,pi,100))
    elseif type == "TiltedPlane"
        obj = exp_ikx_col((N,N); shift_by=(N,N).*angle);
        obj = obj./obj[1];
    else
        print("No such object found")
        return 0
    end
    obj = reshape(obj, (size(obj)...,1))
    return obj
end

function generate_object_3d(;basetype = Float32, type="Dots",samp, Ndots=1, res_dist=1, size_xyz=(100,100,100), dz=100)
    sz = size_xyz;
    CT = Complex{basetype}
    if type == "Dots"
        szd = sz.-(3,3,0)
        idxyz = [(rand(1:szd[1]), rand(1:szd[2]), rand(1:szd[3])) for _ in 1:Ndots]
        dots = delta(szd; offset=idxyz); # possibility of reoccuring Indizes
        obj = select_region(dots, new_size=sz);
    elseif type == "Blobs"
        szd = sz.-(res_dist*2,res_dist*2,res_dist*2)
        idxyz = [(rand(res_dist:szd[1]), rand(res_dist:szd[2]), rand(res_dist:szd[3])) for _ in 1:Ndots]
        dots = disc(sz, res_dist, offset=idxyz); # possibility of reoccuring Indizes
        obj = select_region(dots, new_size=sz);
    elseif type == "Planes"
        obj = zeros(CT,sz);
        plane = ones(round(Int,sz[1]/2),sz[2],1); 
        obj[1:round(Int,sz[1]/2),:,round(Int,sz[3]/4)] = plane .*exp.(-1im.*pi/4);
        obj[round(Int,sz[1]/2)+1:sz[1],:,round(Int,3*sz[3]/4)] = plane .*exp.(-1im.*pi/2);;
        # obj=obj
    elseif type == "Discs"
        obj = zeros(CT,sz);
        cent = round.(Int, sz[1:2]./2);
        radius = (6,6);
        disc1 = disc(sz[1:2], radius, offset= cent.-(12,0)); #disc(sz[1:2], round.(Int,sz[1:2]./8), offset=(sz[1]/4,sz[2]/2));
        disc2 = disc(sz[1:2], radius, offset=cent.+(12,0));#disc(sz[1:2], round.(Int,sz[1:2]./8), offset=(3*sz[1]/4,sz[2]/2));
        obj[:,:,1] = disc1
        obj[:,:,dz] = disc2
        # disc2 = disc(sz[1:2], round.(Int,sz[1:2]./8), offset=(sz[1]/2,sz[2]/2));
        # obj[:,:,dz] = disc2
        # obj=obj
    elseif  type == "roughDisc"
        obj = zeros(CT,sz);
        disc1 = disc(sz[1:2], round.(Int,sz[1:2]./8), offset=(sz[1]/4,sz[2]/2));
        disc2 = disc(sz[1:2], round.(Int,sz[1:2]./8), offset=(3*sz[1]/4,sz[2]/2));
        nz= round(Int, 0.5/samp[3])
        for x in 1:sz[1]
            for y in 1:sz[2]
                obj[x, y, rand(1:nz)] = disc1[x,y]
                obj[x, y, rand(1:nz)+dz-nz] = disc2[x,y]
            end
        end
    elseif type == "roughPlane"
        obj = zeros(CT,sz);
        obj[:,:,1] = ift(pup);
    elseif type == "resolution_dots"
        obj = zeros(CT,sz);
        depth = [dz]#[1, round(Int, sz[3]/2),sz[3]];
        lat = [round(Int, sz[1]/2)-12, round(Int, sz[1]/2), round(Int, sz[1]/2)+12,];
        for idx in range(1, length(depth))
            obj[lat[2],lat[2],depth[idx]] = 1;
            #obj[lat[idx]+res_dist,lat[idx],depth[idx]] = 1;
        end
    end

    return obj
end
