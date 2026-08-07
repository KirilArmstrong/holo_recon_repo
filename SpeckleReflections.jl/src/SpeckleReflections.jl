module SpeckleReflections
using FourierTools
using NDTools
using SeparableFunctions
using SpecialFunctions # for jinc
using Noise # for poisson noise
using MultivariateStats # for svd?
using Random # for rand! in do_simulation

export get_settings, get_settings_meas
export do_simulation, apply_noise
# export get_propagator_mask_3d
export save_all, save_translations, load_all, load_data, ImageSettings, Speckle
export CalibMode, CalibConj, CalibDivide, CalibPhase
export main_reconstruction, apply_translation, get_rel_ref, rec_holo_shift, gen_ref, get_holo_params

include("utils_rec.jl")
include("reconstruction.jl")
include("utils_gen_holo.jl")
include("simulate_data.jl")

end # module SpeckleReflections
