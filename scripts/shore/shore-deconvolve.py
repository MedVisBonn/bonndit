#! /usr/bin/python

import nibabel as nib
# from dipy.reconst.shore import shore_matrix
from dipy.core.gradients import GradientTable, gradient_table
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm
from dipy.io import read_bvals_bvecs
from scipy.special import genlaguerre, gamma
from math import factorial
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import nrrd
import cvxopt
import os
import multiprocessing as mp

import helper.params as params
import tensor.esh
from tensor.esh import esh_to_sym
from tensor.esh import make_kernel_rank1, make_kernel_delta
import dwmri.shore as shore
import tensor.tensor4 as T4

import argparse


# TODO: What about these comments?
# This is now another command-line parameter:
# constrainPos = 'nonneg' # nonnegativity in 300 dirs, a la Tournier
# constrainPos = 'posdef' # positive definiteness
# constrainPos = None


# TODO: Import from somewhere else?
constrainDirs = np.array([2.2772071646, -0.6482608862,
                          2.7114650294, 0.5740959930,
                          0.5633600099, -0.1078841462,
                          1.1281546833, -0.3089940934,
                          2.3784005066, 1.2666118236,
                          1.9559299229, -0.9636440169,
                          1.5710840993, 1.6404386525,
                          2.4564557176, 0.8001964639,
                          1.2478307646, -1.7269168797,
                          0.9208658102, -0.5885453376,
                          2.6812196443, 1.3444029061,
                          0.3448547716, 0.1808514789,
                          2.5309603558, -1.8774895261,
                          2.2136152412, 0.8249771664,
                          2.7968888383, -2.1424848229,
                          0.3268808293, -2.3283121825,
                          2.6584116817, -1.8564466843,
                          1.6342366826, 1.2580881036,
                          0.9283455301, -0.8319778736,
                          1.2808743425, 1.4459013734,
                          0.5199465955, -2.9761731892,
                          0.9853863521, 1.3044279414,
                          0.9668432628, 2.6506273256,
                          0.8681268702, 0.8214782650,
                          0.5273272064, -1.1895059684,
                          2.5572591167, -0.2101715328,
                          2.1008938695, 2.4002379359,
                          1.8775728803, -2.1363852817,
                          0.6752882103, -1.1802073414,
                          2.5272513924, -0.4888308196,
                          2.0491031367, -2.5439962983,
                          0.7751907088, -1.3162165547,
                          1.2924412611, -1.6017206170,
                          1.7170314006, 0.2154868138,
                          1.5775002260, -0.3053038988,
                          1.2225145059, -0.4634301634,
                          2.5322601329, 1.2510468494,
                          1.0243547530, -1.0581076629,
                          0.9645353650, -2.0462732378,
                          0.7890717881, -1.5108137715,
                          1.1693998045, -2.9966262043,
                          2.5817205860, 0.6921208749,
                          0.4359258807, -2.1100786511,
                          2.2167853027, 1.5287569534,
                          1.2992739587, 0.4126497111,
                          1.6971955779, 0.8983874599,
                          1.1255469499, -1.8081054873,
                          1.5839884830, 2.4239096393,
                          0.7559921493, 2.3375710948,
                          0.2350031147, 1.7324517141,
                          1.6958106884, 2.0659195840,
                          0.0724108214, 0.3727452896,
                          1.3142590141, -1.2331938452,
                          2.2993389516, -1.8723085402,
                          1.1996928782, -2.3946878322,
                          1.7435322571, 2.3865164241,
                          1.0391627898, -1.4478758455,
                          0.8175526410, -0.7464017400,
                          1.8632074760, -0.7654562295,
                          1.7995931957, 0.4033142331,
                          0.3557950997, -2.8667227586,
                          0.7510611480, -0.9834789314,
                          0.9465801389, -2.6871726818,
                          0.6908742165, -0.7288664584,
                          2.9506113666, 0.2909962831,
                          2.2683867157, -2.5591960186,
                          0.8761467266, 2.2551072751,
                          0.9883796891, -0.2543181653,
                          1.0749869962, 1.0330201999,
                          2.9334942965, -3.0668409824,
                          2.0874930055, 0.8203868281,
                          0.8281533832, -2.2233359070,
                          2.5355102802, -0.8300569282,
                          1.9330358837, -2.6161833953,
                          1.4329486584, 2.8568164918,
                          1.2981042441, 0.0559570797,
                          0.8887131168, 2.0664687218,
                          1.5485849456, -0.1092228905,
                          1.2094135660, -1.3391956901,
                          0.9911168800, 1.5213245683,
                          2.6852499473, -3.1221893166,
                          2.2136635716, 0.6279777186,
                          1.2177679834, -1.4740297740,
                          0.2155847919, 2.5363569291,
                          2.5472603484, -2.6198545559,
                          1.9973489278, -0.8294339461,
                          2.0177564721, -1.4500539404,
                          2.8160733839, 1.4599118692,
                          1.0241797764, 2.7977517663,
                          2.2611454053, 3.0309866790,
                          2.0815414771, 2.7078869334,
                          2.0628660808, 0.6510347016,
                          2.4439644253, -1.3636815246,
                          2.3451096114, 2.9047244749,
                          1.3549004937, 2.2646806121,
                          1.2407294080, -0.7987387487,
                          1.7681312533, 2.6418474867,
                          0.3395193609, -0.2949656363,
                          0.4749094232, 2.3076445572,
                          1.5464341264, 0.2878487345,
                          1.6966888759, -0.9758573985,
                          1.4115717884, -2.5975049679,
                          1.6719101052, 0.3524429745,
                          1.0771878339, -2.6518787779,
                          2.6773036886, -0.1017303425,
                          1.0027463250, -1.2435954997,
                          0.8007365688, -1.7034165032,
                          1.2870754175, 0.2507802714,
                          1.8977092435, 0.6009311662,
                          2.2385229917, -2.9146420392,
                          2.5385139131, 2.6590421095,
                          1.2103536577, -0.9505167141,
                          1.1403458513, -2.8146995856,
                          1.0859659246, -1.9538606409,
                          0.8442715790, -2.0174118847,
                          1.5624477164, 2.5896699442,
                          1.1298148177, 3.1349445008,
                          1.5205504621, 0.0445319210,
                          0.3604084232, 2.6350214372,
                          1.2978627660, 1.2578882944,
                          1.6852603262, 0.0086386935,
                          1.1755119826, 0.1555524101,
                          1.3137879214, -1.0790591340,
                          0.3541527242, 1.3740480443,
                          1.9804128274, -2.8101777399,
                          2.3129445409, -0.4347682937,
                          2.2898972217, 0.1545912397,
                          2.5894124099, 2.4390398051,
                          2.0378010123, -2.3844188572,
                          1.8382595542, -0.3078387594,
                          1.8835338376, 1.2581348948,
                          0.7067296776, -2.6208304131,
                          2.1463949984, -0.6803243813,
                          2.0185595160, 1.5658995459,
                          1.9715056140, 2.4666829899,
                          2.4485838052, 2.7957826489,
                          2.2636146780, 0.3278965084,
                          1.3934233469, 1.3650102672,
                          2.1105298717, -2.8560221721,
                          2.1172118789, 0.2422024175,
                          1.7189044293, -0.5976106113,
                          0.8806370966, 1.0135267101,
                          1.5800040148, 1.9633111342,
                          0.0888151820, 2.3738247068,
                          0.7486349035, 1.1180020528,
                          1.4490011822, -1.2545989788,
                          1.4603242620, 1.8941025654,
                          1.2191040953, -2.6818073437,
                          2.3044222194, -0.2184845591,
                          0.9699035568, 0.6933685097,
                          0.5449304672, 1.6227635896,
                          2.5134241310, 1.7161990133,
                          2.4109522170, -2.2351497662,
                          0.9918237323, 1.7044406753,
                          1.1261815667, 2.6151125709,
                          1.5909168820, -1.0431420739,
                          1.4994906635, -0.4187335183,
                          0.7161739404, 2.8422878778,
                          0.1895377679, 1.0513415957,
                          1.5449465420, -2.9924433982,
                          2.2494345852, -2.7331777935,
                          1.3600137477, 0.5538180688,
                          0.5929395233, 0.2230414962,
                          1.9712850161, 2.0460292407,
                          1.1632657555, 1.1429305367,
                          0.9999219739, 1.9107118783,
                          1.4611790295, 3.0009547083,
                          1.8774189818, 2.8629027933,
                          2.1811073747, -0.1841491637,
                          1.8476298499, 0.9851651760,
                          0.8731631619, 1.8416090605,
                          1.3016236243, 1.7783513475,
                          2.4379920944, -1.0051471493,
                          1.1619860898, 2.0255574978,
                          1.3864813852, -1.7969455039,
                          0.5927566743, 1.9486374477,
                          1.2774608065, -2.8819479772,
                          1.9632480616, -0.3966920900,
                          1.6105969054, -1.6283029008,
                          1.5764092693, -2.1839106486,
                          1.4119511763, 0.1546389389,
                          1.3203174434, -2.3172740964,
                          1.4872877781, 0.6990971456,
                          1.9643768459, -2.2434365857,
                          2.4992250090, 1.4821462215,
                          1.5252404395, 0.8246523336,
                          1.6022032060, -1.7667795768,
                          2.7494715945, 1.9427309549,
                          1.7477693117, -1.6019650187,
                          1.2329081023, 0.7269229292,
                          0.5836361438, 1.0254728075,
                          1.1343572303, 2.4660398759,
                          1.8504011945, 2.5221460097,
                          1.4535036305, -0.6188576746,
                          2.9241439419, 1.0697051597,
                          2.0595307219, -0.1407783208,
                          2.8297839755, -2.5559195867,
                          2.1191388057, 3.0641480089,
                          1.6720243255, -2.0499015418,
                          2.4501307479, -1.5650336570,
                          1.8225787957, -0.4671097416,
                          1.6286389907, -1.3629361431,
                          0.9459522327, -1.8376270818,
                          0.7648321226, 1.9619979328,
                          1.2764342516, 2.5273356232,
                          0.8976704828, -0.3913651955,
                          1.0426699193, 0.4559896570,
                          0.3274939739, 2.1650883886,
                          1.9669440371, 3.1275539257,
                          2.5749428201, 0.9862207672,
                          1.9812507049, 2.9837549988,
                          1.0012854227, -3.0724719945,
                          1.0672616768, -2.1363045154,
                          3.0457856511, 1.4098971236,
                          2.5522031342, -2.3622709891,
                          2.2532872520, -3.0812438397,
                          1.2711870808, 3.1231757883,
                          0.4039509109, 1.7041026404,
                          2.6543921298, 1.6342487116,
                          0.8489410505, 1.6569995325,
                          1.7696816461, 2.7911422705,
                          0.9154480841, -1.3915079670,
                          1.6040472013, 0.4709988213,
                          1.4265057561, -1.5592400365,
                          1.0893387083, 1.4049101249,
                          1.9759519190, -1.8502877534,
                          2.9621988855, 2.4541271500,
                          2.1689428719, -1.9968178735,
                          1.7642971626, -1.1525957376,
                          2.5751504898, 0.4142108645,
                          2.6530537930, -0.5064188991,
                          1.7999515337, -2.2990131037,
                          2.3941759689, -2.8557617427,
                          2.2535693296, 2.1439830278,
                          1.6846807194, 2.9209846571,
                          0.7382570602, 0.0922318586,
                          2.4042537738, -2.4503516822,
                          1.4649710875, 0.4571888468,
                          1.2141902154, -2.0340268992,
                          1.0299569602, 2.0972938510,
                          1.6621642216, 2.2197113617,
                          0.8678554969, 3.1203719256,
                          0.4753274237, -0.3757162595,
                          2.4077222978, -2.6620123737,
                          1.3730937025, -1.9654521695,
                          1.4697078981, -1.6912331794,
                          1.8809852771, -1.2351128981,
                          0.4692400052, 0.4053613272,
                          1.6579237940, 1.7530156462,
                          1.9676653036, 0.9117105707,
                          1.8266152035, -0.1461543369,
                          2.6989094391, 2.3035284969,
                          1.3571571594, 0.6888561424,
                          2.0658000779, 2.2413123295,
                          1.5683240049, 0.6010574344,
                          1.1765977333, 2.9041194051,
                          1.8067294178, 0.1050824512,
                          1.7208158534, 1.0428311150,
                          1.1313093947, -1.2361243744,
                          1.8203987060, 3.0029699946,
                          0.7058881800, 3.0865748809,
                          2.4719613757, 0.1426867622,
                          1.3716288116, -2.4664168546,
                          1.6756412089, -0.4403529998,
                          2.4209434911, -0.6025113462,
                          0.6052354760, -0.9542185070,
                          1.4112778182, -0.0366702974,
                          1.7259581872, -2.1860343166,
                          2.0756259044, 2.5653215508,
                          1.0090450900, 2.2753689413,
                          2.4265299792, -1.7625663908,
                          2.2872175920, -1.6731935309,
                          1.8830028571, -1.5039753912,
                          1.5961043963, -0.8537870849,
                          0.7369440127, -2.8156894944,
                          1.1629834918, 1.5281067738,
                          2.1311100089, -2.2508019270,
                          1.4168886826, 0.3120066873,
                          1.7917939423, 2.2386960233,
                          0.8703355340, -1.1763224168,
                          1.1265410680, 1.8573949336,
                          0.4531783805, 0.7835843216,
                          1.7896425814, 1.7226732920,
                          1.7055049433, -0.7412315656,
                          0.3213602734, -0.7207416444,
                          2.1010599608, 1.4483985022,
                          2.4530032202, 3.0527884489,
                          2.3438260305, 0.6638319978,
                          2.4339069547, 1.0354533941,
                          1.3361637433, 1.1251935851,
                          2.8282488475, -0.1128235181,
                          1.4765580666, 1.2385658374,
                          1.5380786675, -2.0263797600,
                          1.4843165483, -2.3901331414,
                          1.7324232033, -1.4578947160,
                          1.8324655747, -1.0357216964,
                          0.7820232601, -0.5254536050,
                          0.2532850207, -1.2659842408,
                          1.0434577938, 0.0960065837,
                          0.4635907191, 1.9930757545]).reshape((300, 2))


def signal_to_kernel(signal):
    # rank-1 sh
    T = T4.power(np.array([0, 0, 1]))
    sh = tensor.esh.sym_to_esh(T)
    # print sh

    # Kernel_ln
    kernel = np.zeros((9, 9))
    kernel[0, 0] = signal[0] / sh[0]
    kernel[0, 1] = signal[1] / sh[0]
    kernel[0, 2] = signal[2] / sh[0]
    if len(signal) > 3:
        kernel[2, 2] = signal[3] / sh[3]
        kernel[2, 3] = signal[4] / sh[3]
    if len(signal) > 5:
        kernel[4, 4] = signal[5] / sh[10]
        # print kernel
    return kernel


def signal_to_delta_kernel(signal, order):
    deltash = tensor.esh.eval_basis(order, 0, 0)
    # Kernel_ln
    kernel = np.zeros((order + 1, order + 1))
    counter = 0
    ccounter = 0
    for l in range(0, order + 1, 2):
        for n in range(int((order - l) / 2) + 1):
            kernel[l, l + n] = signal[counter] / deltash[ccounter]
            counter += 1
        ccounter += 2 * l + 3
    # print kernel
    return kernel


# now uses a Quadratic Cone Program to do it in one shot
def deconvolve_posdef(P, q, G, h, init):
    verbose = False
    # first two are orthant constraints, rest positive definiteness
    dims = {'l': 2, 'q': [], 's': [6]}

    # This init stuff is a HACK. It empirically removes some isolated failure cases
    # first, allow it to use its own initialization
    try:
        sol = cvxopt.solvers.coneqp(P, q, G, h, dims)
    except Exception, e:
        print "error-----------", e
        return np.zeros(17)
    if sol['status'] != 'optimal':
        # try again with our initialization
        try:
            sol = cvxopt.solvers.coneqp(P, q, G, h, dims, initvals={'x': init})
        except Exception, e:
            print "error-----------", e
            return np.zeros(17)
        if sol['status'] != 'optimal':
            print 'Optimization unsuccessful.', sol
    c = np.array(sol['x'])[:, 0]
    return c


def call_optimize(i):
    return optimize(i)

class KernelNotSupportedException(Exception):
    pass


def main():
    os.environ['OMP_NUM_THREADS'] = '1'  # visible in this process + all children

    parser = argparse.ArgumentParser(
        description='Calculate the deconvolution of a diffusion MRI signal.')
    parser.add_argument('-i', '--indir', required=True, help='Path to the folder containing all required input files.')
    parser.add_argument('-o', '--outdir', required=True, help='Folder in which the output will be saved. This folder'
                                                              ' needs to already contain the output of "shore-response.py" ')
    parser.add_argument('-c', '--constraint', default='posdef', choices=['posdef', 'nonneg', 'none'],
                        help='Choose a constraint')
# TODO: Does this help makes sense?
    parser.add_argument('-d', '--deconv', default='rank1', choices=['rank1', 'delta'], help='The deconvolution rank.')
    parser.add_argument('-p', '--process', default=2, help='The number of parallel processes.')
    parser.add_argument('-s', '--chunksize', type=int, default=1000, help='The number of voxels per chunk for parallelization.')

    args = parser.parse_args()


    indir = args.indir
    if indir[-1] != '/':
        indir += '/'
    outdir = args.outdir
    if outdir[-1] != '/':
        outdir += '/'
    try:
        img = nib.load(indir + 'data.nii')
    except:
        img = nib.load(indir + 'data.nii.gz')
    data = img.get_data()
    affine = img.affine
    # print affine
    NX, NY, NZ = data.shape[0:3]
    bvals, bvecs = read_bvals_bvecs(indir + 'bvals', indir + 'bvecs')

    # we have to bring b vectors into world coordinate system
    # we will use the 3x3 linear transformation part of the affine matrix for this
    linear = affine[0:3, 0:3]
    # according to FSL documentation, we first have to flip the sign of the
    # x coordinate if the matrix determinant is positive
    if np.linalg.det(linear) > 0:
        bvecs[:, 0] = -bvecs[:, 0]
    # now, apply the linear mapping to bvecs and re-normalize
    bvecs = np.dot(bvecs, np.transpose(linear))
    bvecnorm = np.linalg.norm(bvecs, axis=1)
    bvecnorm[bvecnorm == 0] = 1.0  # avoid division by zero
    bvecs = bvecs / bvecnorm[:, None]

    gtab = gradient_table(bvals, bvecs)

    # mask
    try:
        maskimg = nib.load(indir + 'mask.nii.gz')
        mask = maskimg.get_data()
        print "Using provided DTI mask."
    except:
        mask = np.ones((NX, NY, NZ))
        print "No DTI mask found."

    # load response
    response = np.load(outdir + 'response.npz')
    signal_csf = response['csf']
    signal_gm = response['gm']
    signal_wm = response['wm']
    if 'tau' in response.keys():
        shore_tau = response['tau']
    else:
        shore_tau = 1 / (4 * np.pi ** 2)
    if 'zeta' in response.keys():
        shore_zeta = response['zeta']
    else:
        shore_zeta = 700

    # infer order from response length
    if len(signal_wm) == shore.get_kernel_size(4, 4):
        order = 4
    elif len(signal_wm) == shore.get_kernel_size(6, 6):
        order = 6
    elif len(signal_wm) == shore.get_kernel_size(8, 8):
        order = 8
    else:
        raise KernelNotSupportedException('The response length indicates that the order is not 4, 6 or 8')


    deconv = args.deconv

    if deconv == 'rank1' and order > 4:
        raise KernelNotSupportedException('rank-1 kernels only supported for order 4')
    if args.constraint == 'posdef' and order > 4:
        raise KernelNotSupportedException('posdef constraint only supported for order 4')

    # Kernel_ln
    if deconv == 'rank1':
        kernel_csf = signal_to_kernel(signal_csf)
        kernel_gm = signal_to_kernel(signal_gm)
        kernel_wm = signal_to_kernel(signal_wm)
    else:
        kernel_csf = signal_to_delta_kernel(signal_csf, order)
        kernel_gm = signal_to_delta_kernel(signal_gm, order)
        kernel_wm = signal_to_delta_kernel(signal_wm, order)

    # Build matrix that maps ODF+volume fractions to signal

    # in two steps: First, SHORE matrix
    M_shore = shore.matrix(order, order, shore_zeta, gtab, shore_tau)

    # then, convolution
    M_wm = shore.matrix_kernel(kernel_wm, order, order)
    M_gm = shore.matrix_kernel(kernel_gm, order, order)
    M_csf = shore.matrix_kernel(kernel_csf, order, order)
    M = np.hstack((M_wm, M_gm[:, :1], M_csf[:, :1]))

    # now, multiply them together
    M = np.dot(M_shore, M)

    print 'Condition number of M^T M:', np.linalg.cond(np.dot(M.T, M))

# TODO: What about this outcommented section?
    # v=np.diag(np.linalg.inv(np.dot(np.transpose(M),M)))
    # print 'Variance factors:', v, np.mean(v), np.max(v)


    '''
    x=np.zeros(17)
    x[15]=signal_csf[0]/kernel_csf[0][0]
    print 'Prediction for GM:', np.dot(M,x)
    x[15]=0
    x[16]=signal_csf[0]/kernel_csf[0][0]
    print 'Prediction for CSF:', np.dot(M,x)
    x[16]=0
    x[:15]=tensor.esh.sym_to_esh(T4.power(np.array([0,0,1])))
    print 'Prediction for WM in z dir:', np.dot(M,x)
    '''

    if args.constraint == 'nonneg' or args.constraint == 'posdef':
        cvxopt.solvers.options['show_progress'] = False
        # set up QP problem from normal equations
        P = cvxopt.matrix(np.ascontiguousarray(np.dot(M.T, M)))
        # TODO: consider additional Tikhonov regularization
        if args.constraint == 'nonneg':
            # set up non-negativity constraints
            G = np.zeros((302, tensor.esh.LENGTH[order] + 2))
            counter = 0
            for l in range(0, order + 1, 2):
                for m in range(-l, l + 1):
                    G[:300, counter] = -real_sph_harm(m, l,
                                                      constrainDirs[:, 0],
                                                      constrainDirs[:, 1])
                    counter += 1
            # also constrain GM/CSF VFs to be non-negative
            G[300, tensor.esh.LENGTH[order]] = -1
            G[301, tensor.esh.LENGTH[order] + 1] = -1
            h = np.zeros(302)
        else:
            # set up positive definiteness constraints
            G = np.zeros((38, 17))
            # constrain GM/CSF VFs to be non-negative: orthant constraints
            G[0, 15] = -1
            G[1, 16] = -1
            # positive definiteness constraint on ODF
            ind = np.array(T4.TT).reshape(36)
            for i in range(36):
                G[i + 2, :15] = -np.array(tensor.esh.esh2sym_o4)[ind[i], :]
            h = np.zeros(38)
            # initialize with partly GM, CSF, and isotropic ODF
            init = np.zeros(17)
            init[0] = 0.3
            init[1] = 0.3
            init[2] = 0.3 * signal_csf[0] / kernel_csf[0][0]
            init = cvxopt.matrix(np.ascontiguousarray(init))
        G = cvxopt.matrix(np.ascontiguousarray(G))
        h = cvxopt.matrix(np.ascontiguousarray(h))

    print 'Optimizing...'

    space = (NX, NY, NZ)
    NVOXEL = NX * NY * NZ
    # deconvolution
    out = np.zeros((tensor.esh.LENGTH[order] + 1, NVOXEL))
    gmout = np.zeros(NVOXEL)
    wmout = np.zeros(NVOXEL)
    csfout = np.zeros(NVOXEL)
    mask = mask.reshape(-1)
    data = data.reshape((NVOXEL, -1))

    # need an optimize function to enable multi-processing
    global optimize
    def optimize(i):
        out = np.zeros(tensor.esh.LENGTH[order] + 4)
        if mask[i] == 0:
            return out
        S = data[i, :]
        if args.constraint == 'nonneg' or args.constraint == 'posdef':
            q = cvxopt.matrix(np.ascontiguousarray(-1 * np.dot(M.T, S)))
            if args.constraint == 'nonneg':
                sol = cvxopt.solvers.qp(P, q, G, h)
                if sol['status'] != 'optimal':
                    print 'Optimization unsuccessful.'
                c = np.array(sol['x'])[:, 0]
            else:
                c = deconvolve_posdef(P, q, G, h, init)
        else:
            c = np.linalg.lstsq(M, S)[0]
        out[3] = 1
        out[4:] = esh_to_sym(c[:tensor.esh.LENGTH[order]])
        f = kernel_csf[0][0] / signal_csf[0]
        out[0] = c[0] * f  # wm
        out[1] = c[tensor.esh.LENGTH[order]] * f  # gm
        out[2] = c[tensor.esh.LENGTH[order] + 1] * f  # csf
        return out


    pool = mp.Pool(processes=int(args.process))
    oo = pool.map(call_optimize, range(NVOXEL), args.chunksize)
    #oo = []
    #for i in range(NVOXEL):
    #    oo.append(optimize(i))
    for i in range(NVOXEL):
        wmout[i] = oo[i][0]
        gmout[i] = oo[i][1]
        csfout[i] = oo[i][2]
        out[:, i] = oo[i][3:]

    # output meta data (ODFs)
    meta = {}
    meta['labels'] = ['""' for i in range(4)]
    if order == 4:
        meta['labels'][0] = '"tijk_mask_4o3d_sym"'
    else:
        meta['labels'][0] = '"tijk_mask_8o3d_sym"'
    meta['kinds'] = ['space' for i in range(4)]
    meta['kinds'][0] = '???'
    meta['space'] = 'right-anterior-superior'
    meta['space directions'] = ['none', np.around(affine[0:3, 0], 3), np.around(affine[0:3, 1],3),
                                np.around(affine[0:3, 2],3)]
    meta['space origin'] = affine[0:3, 3].tolist()
    meta['keyvaluepairs'] = {}

    # output
    nrrd.write(outdir + 'odf.nrrd', out.reshape((-1,) + space), meta)

    # output meta data (WM/GM/CSF volume fractions)
    meta = {}
    meta['labels'] = ['""' for i in range(3)]
    meta['kinds'] = ['space' for i in range(3)]
    meta['space'] = 'right-anterior-superior'
    meta['space directions'] = [np.around(affine[0:3, 0],3), np.around(affine[0:3, 1],3), np.around(affine[0:3, 2],3)]
    meta['space origin'] = np.around(affine[0:3, 3],3)
    meta['keyvaluepairs'] = {}
    nrrd.write(outdir + 'vf-gm.nrrd', gmout.reshape(space), meta)
    nrrd.write(outdir + 'vf-wm.nrrd', wmout.reshape(space), meta)
    nrrd.write(outdir + 'vf-csf.nrrd', csfout.reshape(space), meta)


if __name__ == '__main__':
    main()
