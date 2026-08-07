using FileIO, HDF5, Serialization

function save_data(folder, fname, data, fileext=".h5") # save a single file eg. the results of the reconstruction
   h5open(folder*fname*fileext, "w") do fd_holo  # overwrites the current file completely!
      Nimag = length(data)
      for img_num in range(1,Nimag)
         write(fd_holo, string(img_num), data[img_num])
      end
   end
end

function save_all(folder, fname, settings, data_all, illu_all, phase_offset, fileext=".h5")
   h5open(folder*fname*fileext, "w") do fd_holo  # overwrites the current file completely!
        h5open(folder*"illu_"*fname*fileext, "w") do fd_illu
            h5open(folder*"phase_"*fname*fileext, "w") do fd_phase
                Nimag = length(data_all)
                for img_num in range(1,Nimag)
                    write(fd_holo, string(img_num), data_all[img_num])
                    write(fd_illu, string(img_num), illu_all[img_num])
                    # write(fd_ref, string(img_num), ref_all[img_num])
                    write(fd_phase, string(img_num), phase_offset[img_num])
                end
            end # do
        end # do
   end # do
    # h5open(folder*"ref_"*fname*fileext, "w") do fd_ref
   h5open(folder*"settings_"*fname*fileext, "w") do file
      # write(fd_phase, "settings", settings)
      buf = IOBuffer()
      serialize(buf, settings)
      # write(file, "settings", buf)
      attrs(file)["settings"] = take!(buf)
    end # do
   return
end

function save_translations(shifts, fname, fileext=".h5")
    print("\n start saving translation matrix...")
    # ave_img2 = zeros(size(data[:,:,1]))
    h5open(fname*"_translations"*fileext, "w") do file
      for idx in eachindex(shifts)
         # print(idx)
         write(file, string(idx), [shifts[idx]...])
         # ave_img2 .= ave_img2 .+  shift(data[:,:,idx], shifts[idx])
      end
   end
   println("Saved translations to file ", fname*"_translations"*fileext)
end
function load_data_dem2(filename, roi=nothing; T=Float32, preprocess_fct=nothing, range=nothing)
   # brut forced routine to accomodate storage of data as done in setup 2 by Kiril
   """ h5 file structure:
   settings = Dem2Results(load_from_file=str(DATA_PATH/IMG))
      class Dem2Results(Dem2SystemCalculator):
      def __init__(
         self,
         images: np.ndarray | None = None,
         settings: dict | None = None,
         config: dict | None = None,
         b_scan_pattern: Dem2BScanPattern | None = None,
         load_from_file: str = "",
      ):
   log.info(f"Settings: {settings.settings}")
   log.info(f"B Scan Pat: {settings.b_scan_pattern.as_dict()}")
   log.info(f"Images shape: {settings.images.shape}")

   -> for more info visit: https://github.com/myleneamstutz/3d_label_free_dem_2 
   """
 
   data = []
   img_config = []
   scan_config = []
   h5open(filename) do file_local
      data = h5read(filename, keys(file_local)[2])
      img_config = h5read(filename, keys(file_local)[1])
      scan_config = h5read(filename, keys(file_local)[3])
   end
   return data, img_config, scan_config
end

function load_data(filename, roi=nothing; T=Float32, preprocess_fct=nothing, range=nothing)
   Nimag = h5open(filename) do file
      length(keys(file))
   end
   range = isnothing(range) ? (1:Nimag) : range
   println("reading $(range) of $(Nimag) datasets")
   if isnothing(roi)
      if isnothing(preprocess_fct)
         return [T.(h5read(filename,string(d))) for d in range]; # load all the data
      else
         return [preprocess_fct(T.(h5read(filename,string(d)))) for d in range]; # load all the data
      end
   else
      if isnothing(preprocess_fct)
         return [T.(h5read(filename,string(d))[roi...]) for d in range]; # load all the data
      else
         return [preprocess_fct(T.(h5read(filename,string(d))[roi...])) for d in range]; # load all the data
      end
   end
end

function convert_hdf5(folder, fname, fileext=".h5"; T=UInt8)
    all_keys = h5open(folder*fname*fileext) do file
        keys(file)
    end
    Nimag = length(all_keys)
    println("reading $(Nimag) datasets")

    h5open(folder*fname*"_converted"*fileext, "w") do file
        for key in all_keys
            println("Converting: $(string(key))")
            data = T.(h5read(folder*fname*fileext, key)); # load the key
            write(file, key, data)   
        end
    end
end

function load_all(folder, fname, fileext=".h5"; roi = nothing, T=Float32, preprocess_fct=nothing, range=nothing)
   # Nimag = h5open(folder*fname*fileext) do file
   #    length(keys(file))
   # end
   # println("reading $(Nimag) datasets")
   data_filename = folder*fname*fileext;
   data_all = load_data(data_filename,roi; T=T, preprocess_fct=preprocess_fct, range=range)
   Nimag = length(data_all)
   range = isnothing(range) ? (1:Nimag) : range

   illu_filename = folder*"illu_"*fname*fileext;
   illu_all = load_data(illu_filename,roi; T=T, preprocess_fct=preprocess_fct)
   # [h5read(folder*fname*fileext,string(d)) for d in 1:Nimag]; # load all the data
   # illu_all = [h5read(folder*"illu_"*fname*fileext,string(d)) for d in 1:Nimag]; # load all the data
   # ref_all = [h5read(folder*"ref_"*fname*fileext,string(d)) for d in 1:Nimag]; # load all the data
   phase_offset = [h5read(folder*"phase_"*fname*fileext,string(d)) for d in range]; # load all the data
   settings = h5open(folder*"settings_"*fname*fileext,"r") do file
      attrs(file)["settings"]
      deserialize(IOBuffer(attrs(file)["settings"]))
   end
   
   return settings, data_all, illu_all, phase_offset
end

function show_keys(fname, fileext=".h5")
   Nimag = h5open(fname*fileext) do file
      @show length(keys(file))
      @show keys(file)
   end
end

function apply_translation(data, translation_px, T=Float32; verbose=true)
    N = size(data, 2)
    # back translate reconstructed sampel images according translation matrix
    t = Translation(translation_px[1:2]...)
    # t1 = translation_px[1]; #-N/4;
    # t2 = translation_px[2]; #-N/4;
    # t = Translation(t1*2, t2*2)
    (verbose) && println("shifting image my $(translation_px[1:2])")
    data = T.(data)
    return warp(data .- sum(data)/length(data), t, indices_spatial(data), fillvalue=0)
end

"""
    read_hdf5(filepath)

Simple Method to read in data from a hdf5 file, when dictionary is not known
+ filepath: path that points to the hdf5 file
"""
function read_hdf5(filepath; Key_type=0)
    q = load(filepath)
    res = []
    if Key_type==0
      ks = string.(sort(parse.(Int32, String.(keys(q)))[:])) #sort(String.(keys(q)))
    else
      ks = string.(sort(parse.(Float64, String.(keys(q)))[:])) # MA for negative numbers in keys
    end

    for k in ks
      push!(res, q[k])
    end
    w = cat(res..., dims=3)
    return w
end

"""
read_illu_slice(folder,fname, crop_params)

Simple Method to read in the illumination hologram from a hdf5 file and crop to desiered region and then return it.
+ folder: path that points to the folder which hold the hdf5 file
+ fname: name of the hdf5 file that contains the illumination hologramm
+ crop_params: CropParams struct that holds the peremeters for cropping the real space Image before processing
"""
function read_illu_slice(filepath, crop_params; slice_nbr=1)
   # read in illumination data
   @time data_illu = h5read(filepath, string(slice_nbr)) # read in hologramm of illumination holo
   # crop illumination to region of interest
   @time raw_illu = select_region_view(data_illu, new_size=crop_params.crop_size, center=crop_params.crop_center)
   return raw_illu
end
