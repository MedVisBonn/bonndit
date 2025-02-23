#cython: language_level=3, boundscheck=False, wraparound=False, warn.unused=True, warn.unused_args=True
# warn.unused_results=True, profile=True
import sys
import nrrd
sys.path.append('.')
from .alignedDirection cimport  TractSegGetter, Gaussian, Laplacian, ScalarOld, ScalarNew, Probabilities, Deterministic,Deterministic2, WatsonDirGetter, BinghamDirGetter
from .ItoW cimport Trafo
from .stopping cimport Validator
from .integration cimport  Euler, Integration, EulerUKF, RungeKutta
from .interpolation cimport  TomReg, DeepReg, FACT, Trilinear, Interpolation, UKFFodf, UKFFodfAlt, UKFMultiTensor, UKFBinghamAlt, TrilinearFODF, UKFBingham, UKFWatson, UKFWatsonAlt, UKFBinghamQuatAlt, DeepLearned
from bonndit.utilc.cython_helpers cimport mult_with_scalar, sum_c, sum_c_int, set_zero_vector, sub_vectors, \
    angle_deg, norm
import numpy as np
from tqdm import tqdm
from bonndit.utilc.blas_lapack cimport * 


cdef void tracking(double[:,:,:,:] paths, double[:] seed,
                   Interpolation interpolate,
              Integration integrate, Trafo trafo, Validator validator, int max_track_length, int save_steps,
                       int samples, double[:,:,:,:] features, features_save, int minlen, runge_kutta=1):
    # nogil except *:
    """
        Initializes the tracking for one seed in both directions.
    @param paths:
    @param old_dir:
    @param seed:
    @param seed_shape:
    @param interpolate:
    @param integrate:
    @param trafo:
    @param validator:
    @param max_track_length:
    @param samples:
    @param features:
    """
    cdef int k=0, j, l, m, u
    cdef bint skip = False

    for j in range(samples):
        k=0
        while True:
            k+=1
            #print(5)
            # set zero inclusion check
            set_zero_vector(validator.ROIIn.inclusion_check)
            validator.ROIEnd.checker_reset()
            if seed.shape[0] == 3:
                skip = interpolate.main_dir(paths[j, 0, 0])
                if not skip:
                    break
                cblas_dcopy(3, &interpolate.next_dir[0], 1, &integrate.old_dir[0], 1)
            else:
                #skip = interpolate.check_point(seed[:3])
                #if not skip:
                    #print(8)
                #   break
                integrate.old_dir = seed[3:]
                integrate.first_dir = seed[3:]
            validator.ROIEnd.this_run = -1
            status1, m = forward_tracking(paths[j,:,0, :], interpolate, integrate, trafo, validator, max_track_length, save_steps,
                             features[j,:,0, :], features_save, runge_kutta)

            if seed.shape[0] == 3:
                skip = interpolate.main_dir(paths[j, 0, 1])
                if not skip:
                    break
                mult_with_scalar(integrate.old_dir, -1.0 ,interpolate.next_dir)
            else:
                mult_with_scalar(integrate.old_dir, -1.0 ,seed[3:])

            validator.ROIEnd.this_run = -1
            status2, l = forward_tracking(paths[j,:,1,:], interpolate, integrate, trafo, validator, max_track_length, save_steps,
                             features[j,:,1, :], features_save, runge_kutta)
            # if not found bot regions of interest delete path.
            if validator.ROIIn.included_checker() or not status1 or not status2 or (l+m)*integrate.stepsize < minlen or not validator.ROIEnd.check():
                #print('I reset')
                validator.set_path_zero(paths[j,:,1,:], features[j,:,1, :])
                validator.set_path_zero(paths[j, :, 0, :], features[j, :, 0, :])
                for u in range(3):
                    paths[j, 0, 0,u] = seed[u]
                    paths[j, 0, 1,u] = seed[u]
                if 'seedpoint' in list(features_save.keys()):
                    features[j, 0, 0, features_save['seedpoint']] = 1
                    features[j, 0, 1, features_save['seedpoint']] = 1
            else:
                break
            if k==1:
                break

cdef forward_tracking(double[:,:] paths,  Interpolation interpolate,
                       Integration integrate, Trafo trafo, Validator validator, int max_track_length, int save_steps, double[:,:] features, feature_save, int runge_kutta=1): # nogil except *:

    """
        This function do the tracking into one direction.
    @param paths: empty path array to save the streamline points.
    @param interpolate: Interpolation object
    @param integrate: Integration object
    @param trafo: Trafo object
    @param validator: Validator object
    @param max_track_length:
    @param features: empty feature array. To save informations to the streamline
    """
    # check wm volume
    cdef int k, counter=-1
    # thousand is max length for pathway
    interpolate.prob.old_fa = 1
    validator.WM.reset()
    cdef double c = 0
    cdef float con = 0
    for k in range((max_track_length-1)):
        # validate index and wm density.
        counter+=1
        if validator.index_checker(paths[k]):
            set_zero_vector(paths[k])
            break
        # check if neigh is wm.
        con = validator.WM.wm_checker(paths[k])
        if con == 0:
            # If not wm. Set all zero until wm
            for l in range(k):
                con = validator.WM.wm_checker_ex(paths[k - l])
                if con == 0:
                    set_zero_vector(paths[k - l])
                else:
                    break
            break
        elif con > 0:
            if 'wm' in list(feature_save.keys()):
                features[k,feature_save['wm']] = con

        # find matching directions
        if sum_c(integrate.old_dir) == 0:
            set_zero_vector(paths[k])
            set_zero_vector(features[k])

            break
        if 'reg0' in list(feature_save.keys()):
            features[k, feature_save['reg0']] = interpolate.reg[0]
        if 'reg1' in list(feature_save.keys()):
            features[k, feature_save['reg1']] = interpolate.reg[1]
        if 'reg2' in list(feature_save.keys()):
            features[k, feature_save['reg2']] = interpolate.reg[2]
        if 'low0' in list(feature_save.keys()):
            features[k, feature_save['low0']] = interpolate.low[0]
        if 'low1' in list(feature_save.keys()):
            features[k, feature_save['low1']] = interpolate.low[1]
        if 'low2' in list(feature_save.keys()):
            features[k, feature_save['low2']] = interpolate.low[2]
        if 'opt0' in list(feature_save.keys()):
            features[k, feature_save['opt0']] = interpolate.opt[0]
        if 'opt1' in list(feature_save.keys()):
            features[k, feature_save['opt1']] = interpolate.opt[1]
        if 'opt2' in list(feature_save.keys()):
            features[k, feature_save['opt2']] = interpolate.opt[2]
        if 'lambda' in list(feature_save.keys()):
            features[k, feature_save['lambda']] = interpolate.selected_lambda
        if 'angle' in list(feature_save.keys()):
            features[k, feature_save['angle']] = interpolate.angle




        if interpolate.interpolate(paths[k], integrate.old_dir, k) != 0:
            break

        # Check next step is valid. If it is: Integrate. else break
        if validator.next_point_checker(interpolate.next_dir):
            set_zero_vector(paths[k])
            break

        if integrate.integrate(interpolate.next_dir, paths[k - counter%runge_kutta], 1 + counter%runge_kutta)!= 0:
            break

        if sum_c(integrate.next_point) == 0:
            break

        # update old dir
        paths[k + 1] = integrate.next_point
        # check if next dir is near region of interest:

        validator.ROIIn.included(paths[k])
        if validator.ROIEx.excluded(paths[k]):
            return False, k
        if validator.Curve.curvature_checker(paths[:k], features[k:k +  1,1]):
            break
            #if not validator.WM.sgm_checker(trafo.point_wtoi):
#
 ##               print(10)
  #              return False, k
        if validator.ROIEnd.end_checker(paths[k], 1):
            break
    return True, k

cpdef tracking_all(vector_field, wm_mask, tracking_parameters, postprocessing, ukf_parameters, trilinear_parameters, logging, saving, tck):
    """
    @param vector_field: Array (4,3,x,y,z)
        Where the first dimension contains the length and direction, the second
        contains the directions.
    @param meta: Dictionary. Keys: space directions and space origin
        meta data of multivectorfield
    @param wm_mask: Array (x,y,z)
        WM Mask to check wm density
    @param seeds: Array (3,r) or (6,r)
        seedpoint array. If first first axis has length 3 only seed point is given.
        If 6 also the initial direcition is given.
    @param integration: String
        Only Euler integration is available.
    @param interpolation: String
        FACT und Trilinear are available
    @param prob: String
        Gaussian, Laplacian, ScalarOld and ScalarNew are implemented
    @param stepsize: double
        Stepsize for Euler integration.
    @param variance: double
        Variance of Probabilistic Selection Method.
    @param samples: int
        Count of samples per seed.
    @param max_track_length: int
        Maximal length of each track.
    @param wmmin: double
        Minimal White Matter density
    @param expectation: double
        Expectation of Probabilistic selection method.
    @return:
    """
    cdef Interpolation interpolate
    cdef Integration integrate
    cdef Trafo trafo
    cdef Probabilities directionGetter
    cdef Validator validator
    #select appropriate model #TODO hier das richtige einfügren
    if tracking_parameters['ukf'] == "Watson" or tracking_parameters['ukf'] == "WatsonAlt":
        directionGetter = WatsonDirGetter(**tracking_parameters)
        #directionGetter.watson_config(vector_field[0], tracking_parameters['maxsamplingangle'], tracking_parameters['maxkappa'], tracking_parameters[])
    elif tracking_parameters['ukf'] == "Bingham" or tracking_parameters['ukf'] == "BinghamAlt" or tracking_parameters['ukf'] == "BinghamQuatAlt":
        directionGetter = BinghamDirGetter(**tracking_parameters)
    elif tracking_parameters['prob'] == "Gaussian":
        directionGetter = Gaussian(0, tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "Laplacian":
        directionGetter = Laplacian(0, tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "ScalarOld":
        directionGetter = ScalarOld(tracking_parameters['expectation'], tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "ScalarNew":
        directionGetter = ScalarNew(tracking_parameters['expectation'], tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "Deterministic":
        directionGetter = Deterministic(tracking_parameters['expectation'], tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "Deterministic2":
        directionGetter = Deterministic2(tracking_parameters['expectation'], tracking_parameters['variance'])
    elif tracking_parameters['prob'] == "TractSeg":
        directionGetter = TractSegGetter(**tracking_parameters)
    else:
        logging.error('Gaussian or Laplacian or Scalar are available so far. ')
        return 0

    trafo = <Trafo> Trafo(np.float64(tracking_parameters['space directions']), np.float64(tracking_parameters['space origin']))
    trafo_matrix = np.zeros((4,4))
    trafo_matrix[:3,:3] = tracking_parameters['space directions']
    trafo_matrix[:3,3] = tracking_parameters['space origin']
    trafo_matrix[3,3] = 1
    validator = Validator(np.array(wm_mask.shape, dtype=np.intc), postprocessing['inclusion'], postprocessing['exclusion'], trafo,  **tracking_parameters)



    cdef int[:] dim = np.array(vector_field.shape, dtype=np.int32)

    if tracking_parameters['ukf'] == "MultiTensor":
        interpolate = UKFMultiTensor(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "LowRank":
        interpolate = UKFFodf(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "LowRankAlt":
        interpolate = UKFFodfAlt(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "WatsonAlt":
        if 'loss' in saving['features']:
            ukf_parameters['store_loss'] = True
        else:
            ukf_parameters['store_loss'] = False
        interpolate = UKFWatsonAlt(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "Watson":
        if 'loss' in saving['features']:
            ukf_parameters['store_loss'] = True
        else:
            ukf_parameters['store_loss'] = False
        interpolate = UKFWatson(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "Bingham":
        if 'loss' in saving['features']:
            ukf_parameters['store_loss'] = True
        else:
            ukf_parameters['store_loss'] = True
        interpolate = UKFBingham(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "BinghamAlt":
        if 'loss' in saving['features']:
            ukf_parameters['store_loss'] = True
        else:
            ukf_parameters['store_loss'] = True
        interpolate = UKFBinghamAlt(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['ukf'] == "BinghamQuatAlt":
        if 'loss' in saving['features']:
            ukf_parameters['store_loss'] = True
        else:
            ukf_parameters['store_loss'] = True
        interpolate = UKFBinghamQuatAlt(vector_field, dim[2:5], directionGetter, **ukf_parameters)
    elif tracking_parameters['interpolation'] == "FACT":
        interpolate = FACT(vector_field, dim[2:5], directionGetter, **tracking_parameters)
    elif tracking_parameters['interpolation'] == "Trilinear":
        interpolate = Trilinear(vector_field, dim[2:5], directionGetter, **tracking_parameters)
    elif tracking_parameters['interpolation'] == "TrilinearFODF":
        interpolate = TrilinearFODF(vector_field, dim[2:5], directionGetter, **trilinear_parameters)
    elif tracking_parameters['interpolation'] == "TractSeg":
        interpolate = DeepReg(vector_field, dim[2:5], directionGetter, **tracking_parameters)
    elif tracking_parameters['interpolation'] == "TOM":
        interpolate = TomReg(vector_field, dim[2:5], directionGetter, **tracking_parameters)
    elif tracking_parameters['interpolation'] == "Learned":
        interpolate = DeepLearned(vector_field, dim[2:5], directionGetter, **tracking_parameters)
    else:
        logging.error('FACT, Triliniear or UKF for MultiTensor and low rank approximation are available so far.')
        return 0

    if tracking_parameters['ukf'] == "MultiTensor" or tracking_parameters['integration'] == "EulerUKF":
        integrate = EulerUKF(tracking_parameters['space directions'], tracking_parameters['space origin'], trafo,
                             float(tracking_parameters['stepsize']))
    elif tracking_parameters['integration'] == "Euler":
        integrate = Euler(tracking_parameters['space directions'], tracking_parameters['space origin'], trafo, float(tracking_parameters['stepsize']))
    elif tracking_parameters['integration'] == "RungeKutta":
        integrate = RungeKutta(tracking_parameters['space directions'], tracking_parameters['space origin'], trafo, float(tracking_parameters['stepsize']), **{'interpolate': interpolate})
    else:
        logging.error('Only Euler is available so far. Hence set Euler as argument.')
        return 0

    cdef int i, j, k, l, m, choice
    print("seed_count", tracking_parameters['seed_count'])
    if tracking_parameters['seed_count'] == 0:
        m = tracking_parameters['seeds'].shape[0]
    else:
        m = tracking_parameters['seed_count']
    # Array to save Polygons
    cdef double[:,:,:,:,:] paths = np.zeros((1, tracking_parameters['samples'], tracking_parameters['max_track_length'], 2, 3),dtype=np.float64)
    # Array to save features belonging to polygons
    cdef double[:,:,:,:,:] features = np.zeros((1, tracking_parameters['samples'], tracking_parameters['max_track_length'], 2, len(saving['features'].keys())),dtype=np.float64)
    # loop through all seeds.
    tracks = []
    tracks_len = []
    k = 0

    for i in tqdm(range(m), disable=not tracking_parameters['verbose']):
        #k = 0 if saving['file'] else k+=1
        #Convert seedpoint
        #trafo.wtoi(seeds[i][:3])
        for j in range(tracking_parameters['samples']):
            #print(1)
            validator.set_path_zero(paths[k,j, :, 1, :], features[k,j, :, 1, :])
            validator.set_path_zero(paths[k,j, :, 0, :], features[k,j, :, 0, :])
            #print(2)
            if tracking_parameters['seed_count'] == 0:
                choice = i
            else:
                choice = np.random.randint(0, len(tracking_parameters['seeds']))

            #print(3)
            for l in range(3):
                paths[k,j, 0, 0,l] = tracking_parameters['seeds'][choice][l]
                paths[k,j, 0, 1,l] = tracking_parameters['seeds'][choice][l]
            for l in range(3):
                paths[k,j, 0, 0,l] +=  np.random.uniform(-0.5,0.5)
                paths[k,j, 0, 1,l] = paths[k,j, 0, 0,l]
            #print(np.array(paths[k,j, 0, 0]))
        if saving['features']['seedpoint'] >= 0:
            features[k,:, 0, 0, saving['features']['seedpoint']] = 1
            features[k,:, 0, 1, saving['features']['seedpoint']] = 1
        #Do the tracking for this seed with the direction
        tracking(paths[k], tracking_parameters['seeds'][choice],  interpolate, integrate, trafo, validator, tracking_parameters['max_track_length'], tracking_parameters['sw_save'], tracking_parameters['samples'], features[k], saving['features'], tracking_parameters['min_len'], tracking_parameters['runge_kutta'])
        # delete all zero arrays.
        #print(k, np.array(paths[k]))

        for j in range(tracking_parameters['samples']):
            feature =  features[0,j,::tracking_parameters['runge_kutta']]
            path = paths[0,j,::tracking_parameters['runge_kutta']]
            path = np.concatenate((path[1:,0][::-1], path[:,1]))
            feature = np.concatenate((feature[1:, 0][::-1], feature[:, 1]))
            #try:
            #print('path', path)
            to_exclude = np.all(path[:,:] == 0, axis=1)
            path = path[~to_exclude]
            feature = feature[~to_exclude]
            if path.size == 0:
                continue
            if path.shape[0]>5:
                path = np.vstack((path[::int(tracking_parameters['sw_save'])], path[len(path)-1][np.newaxis]))
                feature = np.vstack((feature[::int(tracking_parameters['sw_save'])], feature[len(feature) - 1][np.newaxis]))
                feature_to_add = {}
                for key in saving['features'].keys():
                    feature_to_add[key] = feature[..., saving['features'][key]]

                tck.append(path, feature_to_add)
            #except:
            #   pass

    return tracks, tracks_len


