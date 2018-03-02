import numpy as np
import numpy.linalg as la
import math

def parse(data, meta, literal = False):

	n = data.shape[0]
	bmax = float(meta['keyvaluepairs']['DWMRI_b-value'])
	B = np.zeros(n)
	G = np.zeros((n, 3))
	#print meta['keyvaluepairs']['DWMRI_gradient_0000']

	for i in range(n):
		key = 'DWMRI_gradient_{:04}'.format(i)
		try:
			g = meta['keyvaluepairs'][key]
			g = np.array([float(x) for x in g.split(' ')])
			l2 = g[0]**2 + g[1]**2 + g[2]**2
			B[i] = bmax * l2
			if literal:
				G[i,:] = g
			elif l2 > 0:
				G[i,:] = g / math.sqrt(l2)
		except Exception as e:
			print e
			pass
	return G, B

# G must be "literal"
def meta_set_gb(meta, G):
	bmax = float(meta['keyvaluepairs']['DWMRI_b-value'])
	n = len(G)
	
	# remove old gradients
	to_del = []
	for k in meta['keyvaluepairs']:
		if k[:15] == 'DWMRI_gradient_':
			to_del.append(k)
	for k in to_del:
		del meta['keyvaluepairs'][k]

	# create new gradients
	for i in range(n):
		key = 'DWMRI_gradient_{:04}'.format(i)
		meta['keyvaluepairs'][key] = '{:.6f} {:.6f} {:.6f}'.format(G[i,0], G[i,1], G[i,2])
	return meta

def get_s0(data, B):
	s0 = np.zeros(data.shape[1:])
	n = 0
	for i,b in enumerate(B):
		if b > 20:
			continue
		for j,x in np.ndenumerate(s0):
			s0[j] += data[(i,) + j]
		n += 1
	if n == 0:
		raise Exception("no B=0 image found (B<20)")
	return s0 / n

def apply_measurement_frame(G, meta):
	frame = get_measurement_frame(meta)
	for g in G:
		g = np.dot(frame, g)
	return G

def get_measurement_frame(meta):
	frame = []
	for d in meta['space directions']:
		if d != 'none':
			frame += [d]
	if len(frame) != 3:
		raise Exception('invalid measurement frame: ' + meta['space directions'])
	frame = np.array(frame)
	return frame

# Helper function to transform 3D Cartesian coordinates into
# theta (polar angle from positive z) and phi (azimuth from positive x)
# Input vector should be non-zero, but does not need to be normalized
def cart2thetaphi(v):
	r = la.norm(v)
	if r == 0:
		return 0, 0
	z = v[2] / r
	if z > 1:
		theta = 0
	elif z < -1:
		theta = math.pi
	else:
		theta = math.acos(z)
	phi = math.atan2(v[1], v[0])
	return theta, phi

def compute_response(order, bval, md, delta):
	response = np.zeros(order / 2 + 1)

	# these analytical expressions have been derived using sage
	exp1 = math.exp(bval*(delta-md))
	exp2 = math.exp(3*bval*delta)
	exp3 = math.exp(bval*(-md-2*delta))
	erf1 = math.erf(math.sqrt(3*bval*delta))
	sqrtbd = math.sqrt(bval*delta)
	response[0] = math.pi / math.sqrt(3.0)*exp1*erf1/sqrtbd
	if order < 2:
		return response
	response[1] = -0.186338998125 * math.sqrt(math.pi)*\
    (2.0*math.sqrt(3.0*math.pi)*exp1*erf1/sqrtbd +
     (-math.sqrt(math.pi)*exp2*erf1 + 2.0*sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*delta))
	if order < 4:
		return response
	response[2]=0.0104166666667*math.sqrt(math.pi)*\
    (36.0*math.sqrt(3*math.pi)*exp1*erf1/sqrtbd -
     60*(math.sqrt(math.pi)*exp2*erf1 - 2*sqrtbd*math.sqrt(3.0))*
     math.sqrt(3.0)*exp3/(sqrtbd*bval*delta) +
     35.0*(math.sqrt(math.pi)*exp2*erf1- 2.0*(2.0*bval*delta+1.0)*sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*delta*bval*delta))
	if order < 6:
		return response
	response[3] = -0.00312981881551*math.sqrt(math.pi)*\
    (120.0*math.sqrt(3.0*math.pi)*exp1*erf1/sqrtbd -
     420.0*(math.sqrt(math.pi)*exp2*erf1 - 2*sqrtbd*math.sqrt(3.0))*
     math.sqrt(3.0)*exp3/(sqrtbd*bval*delta) +
     630.0*(math.sqrt(math.pi)*exp2*erf1 - 2.0*(2.0*bval*delta+1.0)
            *sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*bval*delta*delta)-
     77.0*(5.0*math.sqrt(math.pi)*exp2*erf1 -
           2.0*(12.0*bval*delta*bval*delta + 10.0*bval*delta + 5.0)
           *sqrtbd*math.sqrt(3.0))*
     math.sqrt(3.0)*exp3/(sqrtbd*bval*delta*bval*delta*bval*delta))
	if order < 8:
		return response
	response[4] = 0.000223692796529*math.sqrt(math.pi)*\
    (1680.0*math.sqrt(3.0*math.pi)*exp1*erf1/sqrtbd -
     10080.0*(math.sqrt(math.pi)*exp2*erf1 - 2.0*sqrtbd*math.sqrt(3.0))*
     math.sqrt(3.0)*exp3/(sqrtbd*bval*delta) +
     27720.0*(math.sqrt(math.pi)*exp2*erf1 -
              2.0*(2.0*bval*delta+1.0)*sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*delta*bval*delta)-
     8008.0*(5.0*math.sqrt(math.pi)*exp2*erf1 -
             2.0*(12.0*bval*delta*bval*delta +
                  10.0*bval*delta + 5.0)*sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*delta*bval*delta*bval*delta) +
     715.0*(35.0*math.sqrt(math.pi)*exp2*erf1 -
            2.0*(72.0*bval*delta*bval*delta*bval*delta+
                 84.0*bval*delta*bval*delta+
                 70.0*bval*delta+35.0)*sqrtbd*math.sqrt(3.0))
     *math.sqrt(3.0)*exp3/(sqrtbd*bval*delta*bval*delta*bval*delta*bval*delta))
#  if (!AIR_EXISTS(response[1])) {
#    fprintf(stderr, "WARNING: "
#            "bval=%f md=%f delta=%f\n"
#            "exp1=%f exp2=%f exp3=%f erf1=%f sqrtbd=%f\n",
#            bval, md, delta,
#            exp1, exp2, exp3, erf1, sqrtbd)
#  }
#}
	return response



def nrrdMetaRemoveAxis(imeta, axis):
	assert(axis == 0) # TODO...
	ometa = imeta.copy()
	if imeta.has_key('space directions'):
		ometa['space directions'] = imeta['space directions'][1:]
	if imeta.has_key('kinds'):
		ometa['kinds'] = imeta['kinds'][1:]
	if imeta.has_key('centerings'):
		ometa['centerings'] = imeta['centerings'][1:]
	if imeta.has_key('data file'):
		ometa.pop('data file', None)
	return ometa


def nrrdMetaAddAxis(imeta, axis):
	assert(axis == 0) # TODO...
	print imeta
	ometa = imeta.copy()
	if imeta.has_key('labels'):
		ometa['labels'] = ['""'] + imeta['labels']
	if imeta.has_key('space directions'):
		ometa['space directions'] = ['none'] + imeta['space directions']
	if imeta.has_key('kinds'):
		ometa['kinds'] = ['???'] + imeta['kinds']
	if imeta.has_key('centerings'):
		ometa['centerings'] = [''] + imeta['centerings']
	if imeta.has_key('data file'):
		ometa.pop('data file', None)
	return ometa
