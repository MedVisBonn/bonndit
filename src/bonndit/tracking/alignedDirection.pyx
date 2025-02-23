#%%cython --annotate
#cython: language_level=3, boundscheck=False,
import cython
from libc.math cimport acos, pi, exp, fabs, cos, pow, tanh, sqrt
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from bonndit.utilc.cython_helpers cimport scalar, clip, mult_with_scalar, sum_c, norm
import numpy as np
from scipy.special import dawsn, gamma, iv
from bonndit.utilc.blas_lapack cimport *


###
# Given a set of vectors should be in shape [n,3] and a vector returns most aligned vector shape [3]

cdef class Probabilities:
    srand(time(NULL))



    def __cinit__(self, double expectation=0, double sigma=9, **kwargs):
        self.sigma = sigma
        self.expectation = expectation
        self.chosen_prob = 0
        self.probability = np.zeros((3,))
        self.angles = np.zeros((3,))
        self.best_fit = np.zeros((3))
        print(hex(id(self.best_fit)))
        self.test_vectors = np.zeros((3, 3))
        print(hex(id(self.test_vectors)))
        self.chosen_angle = 0
        self.old_fa = 1


    cdef void aligned_direction(self, double[:,:] vectors, double[:] direction) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        #calculate angle between direction and possibilities. If angle is bigger than 90 use the opposite direction.
        cdef int i, n = vectors.shape[0]
        cdef double test_angle, min_angle = 180

        for i in range(n):
            if sum_c(direction) == 0:
                self.angles[i] = 0
                continue
            if norm(vectors[i]) != 0 and norm(vectors[i]) == norm(vectors[i]):
                test_angle = clip(scalar(direction, vectors[i])/(norm(direction)*(norm(vectors[i]))), -1,1)
                if test_angle >0 :
                    self.angles[i] = acos(test_angle)/pi*180
                    self.test_vectors[i] = vectors[i]
                elif test_angle <= 0:
                    self.angles[i] = 180 - acos(test_angle)/pi*180
                    mult_with_scalar(self.test_vectors[i], -1, vectors[i])
            else:
                self.angles[i] = 180
                mult_with_scalar(self.test_vectors[i], 0, vectors[i])

    cdef int random_choice(self, double[:] direction) : # nogil  except *:
        """

        @param direction:
        @return:
        """
        cdef double best_choice = rand() / RAND_MAX
        cdef int c_ind = 0
    #   with gil:
    #       print(*self.probability)
        if sum_c(self.probability) != 0:
            mult_with_scalar(self.probability, 1/sum_c(self.probability), self.probability)


            if best_choice < self.probability[0]:
                mult_with_scalar(self.best_fit, 1, self.test_vectors[0])
                self.chosen_prob = self.probability[0]
                self.chosen_angle = self.angles[0]
                c_ind = 0
            elif best_choice < self.probability[0] + self.probability[1]:
                mult_with_scalar(self.best_fit, 1, self.test_vectors[1])
                self.chosen_prob = self.probability[1]
                self.chosen_angle = self.angles[1]
                c_ind = 1
            else:
                mult_with_scalar(self.best_fit, 1, self.test_vectors[2])
                self.chosen_prob = self.probability[2]
                self.chosen_angle = self.angles[2]
                c_ind = 2
            self.old_fa = norm(self.best_fit)
        else:

            mult_with_scalar(self.best_fit, 0, self.test_vectors[2])
            self.chosen_angle = 0
            self.chosen_prob = 0
            c_ind = -1
        #with gil:
        #   print(*self.probability, ' and I chose ', self.chosen_prob, ' where the angle are ', *self.angles, ' I chose ', self.chosen_angle)
        return  c_ind


    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil except *:
        pass


    cdef void calculate_probabilities_sampled(self, double[:,:] vectors, double[:] kappas, double[:] weights, double[:] direction, double[:] point) : # nogil  except *:
        pass

    cdef void calculate_probabilities_sampled_bingham(self, double[:,:] vectors, double[:] old_dir, double[:,:, :] A, double[:,:] l_k_b) : # nogil  except *:
        pass

    cdef void select_next_dir(self, double[:] next_dir, double[:] old_dir):
        pass

cdef class Gaussian(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i
        self.aligned_direction(vectors, direction)
        for i in range(3):
            if self.angles[i] < 50:
                self.probability[i] = exp(-1/2*((self.angles[i] - self.expectation)/self.sigma)**2)
            else:
                self.probability[i] = 0
        self.random_choice(direction)


cdef class Laplacian(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i
        self.aligned_direction(vectors, direction)
        for i in range(3):
            self.probability[i] = 1/2 * exp(- (fabs(self.angles[i] - self.expectation) /
                                                         self.sigma))
        self.random_choice(direction)


cdef class ScalarOld(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i
        cdef double s
        self.aligned_direction(vectors, direction)
        #with gil:
        #   print(*self.angles)
        for i in range(3):
            if sum_c(self.test_vectors[i]) == sum_c(self.test_vectors[i])  and pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2) <= 1/2*pi:
            #   with gil:
            #       print('First angle ' , self.angles[i], pow(cos(pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2)),self.sigma)*norm(self.test_vectors[i]))
                self.probability[i]=pow(cos(pow(self.expectation/pow(2*pi,0.5)*self.angles[i]/180*pi,2)),self.sigma)*exp(-pow(norm(self.test_vectors[i]) - self.old_fa,2)/0.01)


            else:
                self.probability[i] = 0
        self.random_choice(direction)


cdef class ScalarNew(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i
        cdef double s
        self.aligned_direction(vectors, direction)
        #with gil:
        #   print(*self.angles)
        for i in range(3):
            if sum_c(vectors[i]) == sum_c(vectors[i]):
                self.probability[i] = pow(cos(self.angles[i]/180*pi),self.sigma)*norm(self.test_vectors[i])
            else:
                self.probability[i] = 0

        self.random_choice(direction)


cdef class Deterministic2(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i, min_index=0
        cdef double s, min_angle=0
        self.aligned_direction(vectors, direction)
        for i in range(3):
            if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
                if self.angles[i] < min_angle or i==0:
                    min_angle=self.angles[i]
                    min_index=i
        for i in range(3):
            if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
                if self.angles[i] < self.expectation or (self.angles[i] == min_angle and min_angle < self.sigma):
                    self.probability[i] = 1
                else:
                    self.probability[i] = 0
        self.random_choice(direction)

cdef class Deterministic(Probabilities):
    cdef void calculate_probabilities(self, double[:,:] vectors, double[:] direction) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i, min_index=0
        cdef double s, min_angle=0
        self.aligned_direction(vectors, direction)
        for i in range(3):
            if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
                if self.angles[i] < min_angle or i==0:
                    min_angle=self.angles[i]
                    min_index=i

        mult_with_scalar(self.best_fit, 1, self.test_vectors[min_index])
        self.chosen_prob = 0
        self.chosen_angle = self.angles[min_index]


cdef class WatsonDirGetter(Probabilities):
    def __cinit__(self, double expectation=0, double sigma=9, **kwargs):
        super().__init__(expectation, sigma, kwargs)
        self.max_samplingangle = kwargs['max_sampling_angle']
        self.max_kappa = kwargs['max_kappa']
        self.min_kappa = kwargs['min_kappa']
        self.prob_direction = False #kwargs['prob_direction']

    cdef double poly_kummer(self, double kappa) :#nogil  except *:
        return exp(kappa)/sqrt(kappa) * dawsn(sqrt(kappa))

    cdef double poly_watson(self, double[:] x, double[:] mu, double kappa, double scale) :#nogil  except *:
        return 1/scale * exp(kappa * scalar(mu,x)**2)

    # rejection sampling from watson - watson_confidence_interval.ipynb
    cdef void mc_random_direction(self, double[:] direction, double[:] mu, double kappa, double scale) :#nogil  except *:
        cdef double max_val = self.poly_watson(mu, mu, kappa, scale)
        cdef bint accept = False
        cdef double val, cutoff

        while not accept:
            direction[0] = (rand() / RAND_MAX) * 2 - 1
            direction[1] = (rand() / RAND_MAX) * 2 - 1
            direction[2] = (rand() / RAND_MAX) * 2 - 1
            mult_with_scalar(direction,1/norm(direction),direction)
            val = self.poly_watson(direction, mu, kappa, scale)
            cutoff = (rand() / RAND_MAX) * max_val
            if val > cutoff:
                accept = True

    cdef void calculate_probabilities_sampled(self, double[:,:] vectors, double[:] kappas, double[:] weights, double[:] direction, double[:] point) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i, min_index=0
        cdef double s, min_angle=0, norm_of_test, mc_angle = 360
        cdef double kappa_value

        self.aligned_direction(vectors, direction)

        if self.prob_direction:
            for i in range(vectors.shape[0]):
                if sum_c(vectors[i]) == sum_c(vectors[i]):
                    self.probability[i] = pow(cos(self.angles[i]/180*pi),self.sigma)*norm(self.test_vectors[i])
                else:
                    self.probability[i] = 0

            min_index = self.random_choice(direction)
        else:
            for i in range(vectors.shape[0]):
                if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
                    if self.angles[i] < min_angle or i==0:
                        min_angle=self.angles[i]
                        min_index=i

        kappa_value = kappas[min_index]

        # if kappa is to low the tracking is stopped
        if kappa_value < self.min_kappa:
            self.best_fit = np.zeros((3))
            return

        # normalize length of selected peak direction
        norm_of_test = norm(self.test_vectors[min_index])
        mult_with_scalar(self.test_vectors[min_index],1/norm_of_test,self.test_vectors[min_index])

        while mc_angle > self.max_samplingangle:

            M = 4 * pi * self.poly_kummer(min(self.max_kappa,kappa_value))
            self.mc_random_direction(self.best_fit,
                                 self.test_vectors[min_index],
                                 min(self.max_kappa,kappa_value), M)

            # flip direction if > 90:
            if scalar(self.best_fit, self.test_vectors[min_index]) < 0:
                mult_with_scalar(self.best_fit,-1.0,self.best_fit)

            # compute angle between peak direction and sampled one
            if (norm(self.best_fit)*(norm(self.test_vectors[min_index]))) == 0:
                break
            mc_angle = clip(scalar(self.best_fit, self.test_vectors[min_index])/(norm(self.best_fit)*(norm(self.test_vectors[min_index]))), -1,1)
            # convert to degrees
            mc_angle = acos(mc_angle)/pi*180
    #   print(mc_angle)
        # reset to original length
        mult_with_scalar(self.best_fit, norm_of_test, self.best_fit)

        self.chosen_prob = min(self.max_kappa,kappas[min_index])
        self.chosen_angle = self.angles[min_index]

cdef class BinghamDirGetter(Probabilities):
    def __cinit__(self, double expectation=0, double sigma=9, **kwargs):
        super().__init__(expectation, sigma, kwargs)
        self.max_samplingangle = kwargs['max_sampling_angle']
        self.max_kappa = kwargs['max_kappa']
        self.max_beta= kwargs['max_kappa']
        self.min_kappa = kwargs['min_kappa']
        self.prob_direction = False #kwargs['prob_direction']
        self.min_beta = kwargs['min_kappa']
        self.min_lambda = 0.01 #kwargs['min_lambda']
        self.c = np.zeros((3,), dtype=np.float64)


    cdef double bingham(self, double[:] x, double[:] mu, double kappa, double[:,:] A) :#nogil  except *:
        cdef double res
        #print('mu', np.array(x))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans , 1, 3, 3, 1, &x[0], 1, &A[0,0], 3, 0, &self.c[0], 3)
        #print('c', np.array(self.c))
        res = cblas_ddot(3, &self.c[0], 1, &x[0], 1)
        return  exp(kappa * scalar(mu,x)**2 + res)

    # rejection sampling from watson - watson_confidence_interval.ipynb
    cdef void mc_random_direction(self, double[:] direction, double[:] mu, double kappa, double[:,:] A) :#nogil  except *:
        cdef double max_val = self.bingham(mu, mu, kappa, A)
        cdef bint accept = False
        cdef double val, cutoff
        #l = 0
        while not accept:
        #   l +=1
            direction[0] = (rand() / RAND_MAX) * 2 - 1
            direction[1] = (rand() / RAND_MAX) * 2 - 1
            direction[2] = (rand() / RAND_MAX) * 2 - 1
            mult_with_scalar(direction,1/norm(direction),direction)
            val = self.bingham(direction, mu, kappa, A)
            cutoff = (rand() / RAND_MAX) * max_val
            if val > cutoff:
                accept = True

    cdef void calculate_probabilities_sampled_bingham(self, double[:,:] vectors, double[:] old_dir, double[:,:, :] A, double[:,:] l_k_b) : # nogil  except *:
        """

        @param vectors:
        @param direction:
        @return:
        """
        cdef int i, min_index=0
        cdef double s, min_angle=0, norm_of_test, mc_angle = 360
        cdef double kappa_value

        self.aligned_direction(vectors, old_dir)


        for i in range(vectors.shape[0]):
            if sum_c(vectors[i]) == sum_c(vectors[i]) and sum_c(vectors[i])!=0:
                if self.angles[i] < min_angle or i==0:
                    min_angle=self.angles[i]
                    min_index=i
        # if lambda, kappa, beta is too low the trackin)g is stoppe
        #print('11', np.array(l_k_b))
        if l_k_b[min_index, 0] < self.min_lambda:
            #print('lambda_error')
            self.best_fit = np.zeros((3))
            return
        if l_k_b[min_index, 1] < self.min_kappa:
            #print('kambda_error')
            self.best_fit = np.zeros((3))
            return
        else:
            kappa_value = l_k_b[min_index, 1]


        while mc_angle > self.max_samplingangle:
            self.mc_random_direction(self.best_fit,
                                 self.test_vectors[min_index],
                                 min(self.max_kappa,kappa_value), A[min_index])

            # flip direction if > 90:
            if scalar(self.best_fit, self.test_vectors[min_index]) < 0:
                mult_with_scalar(self.best_fit,-1.0,self.best_fit)

            # compute angle between peak direction and sampled one
            mc_angle = clip(scalar(self.best_fit, self.test_vectors[min_index])/(norm(self.best_fit)*(norm(self.test_vectors[min_index]))), -1,1)
            # convert to degrees
            mc_angle = acos(mc_angle)/pi*180
        #print(self.angles[min_index])

        # reset to original length
        mult_with_scalar(self.best_fit, 1, self.best_fit)
        #print(l_k_b[min_index,2]/l_k_b[min_index,1])
        self.chosen_prob = l_k_b[min_index,2]/l_k_b[min_index,1]
        self.chosen_angle = l_k_b[min_index,1]

#
#
cdef class TractSegGetter(WatsonDirGetter):
    def __cinit__(self, double expectation=0, double sigma=9, **kwargs):
        super().__init__(expectation, sigma, kwargs)
 
    cdef void select_next_dir(self, double[:] next_dir, double[:] old_dir):
        cdef int idx = 0
        cdef double min_v = 0,v
        for i in range(3):
            if next_dir[4*i]  == 0:
                continue
            v = abs(cblas_ddot(3, &next_dir[4*i + 1], 1, &old_dir[0], 1))
            if v > min_v:
                idx = i
                min_v = v
        if min_v < 0.7:
            cblas_dscal(3, 0, &self.best_fit[0], 1)
        else:


            #if cblas_ddot(3, &next_dir[0], 1, &old_dir[0], 1) < 0:
            #cblas_dscal(3, -1, &next_dir[0], 1)

            #print(2, np.asarray(old_dir),hex(id(old_dir)), hex(id(self.best_fit)))
            cblas_dcopy(3, &next_dir[4*idx+1], 1, &self.best_fit[0], 1)

            #self.mc_random_direction(self.best_fit, next_dir, self.sigma, 1)
            #print(3, np.asarray(old_dir), np.asarray(next_dir), hex(id(old_dir[0])), hex(id(self.best_fit[0])))
            if cblas_ddot(3, &old_dir[0], 1, &self.best_fit[0], 1) < 0:
                cblas_dscal(3, -1, &self.best_fit[0], 1)
