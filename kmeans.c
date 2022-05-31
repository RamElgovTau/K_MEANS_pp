#define PY_SSIZE_T_CLEAN  /* For all # variants of unit formats (s#, y#, etc.) use Py_ssize_t rather then int */
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/*********************************
* ram elgov 206867517
* mohammad daghash 314811290
**********************************/
static void free_mat(double** m, int rows) {
  int i;
  for (i = 0; i < rows; ++i) {
    free(m[i]);
  }
  free(m);
}
static int is_converged(double *centroids, double *old_centroids, int K, int d, double epsilon) {
  int i, j;
  double norm;
  /*
   * checks if all centroid's norm didn't change more than the value of epsilon. if yes, then it's converged.
   */
  for (i = 0; i < K; i++) {
    norm = 0;
    for (j = 0; j < d; j++) {
      norm += pow(centroids[i * d + j] - old_centroids[i * d + j], 2);
    }
    norm = pow(norm, 0.5);
    if (norm >= epsilon) {
      return 1;
    }
  }
  return 0;
}
static int index_of_closest_cluster(double *x, double *centroids, int K, int d) {
  double min = 0;
  double sum;
  int i, j, index = 0;
  /* minimum initialisation. (the fis  */
  for (i = 0; i < d; i++) {
    min += pow(x[i] - centroids[i], 2);
  }
  /* checks for the rest of the centroids. */
  for (j = 0; j < K; j++) {
    sum = 0;
    for (i = 0; i < d; i++) {
      sum += pow(x[i] - centroids[j * d + i], 2);
    }
    if (sum < min) {
      min = sum;
      index = j;
    }
  }
  return index;
}
static double** read_2d_array_from_python(int rows, int cols, PyObject *py_list);
static PyObject* pass_2d_array_to_python(int rows, int cols, double **arr);
static int run_kmeans(double** data_points, double** centroids_pp, int n, int d, int k, int max_iter, double epsilon) {
//  int iteration_num, valid, i, j, t;
//  double *vectors;
//  double *centroids;
//  double *old_centroids;
//  double *clusters;
//  int *sizeof_clusters;
//  centroids = calloc(k * d, sizeof(double));
//  vectors = calloc(d * n, sizeof(double));
//  data_points = calloc(n, sizeof(double *));
//  old_centroids = calloc(k * d, sizeof(double));
//  clusters = calloc(k * d, sizeof(double));
//  sizeof_clusters = calloc(k, sizeof(int));
//
//  // kmeans++ centroids initialization ------------------------------------------------------
//  t = 0;
//  i = 0;
//  j = 0;
//  while (t < d * k) {
//    if (j == d) {
//      j = 0;
//      ++i;
//    }
//    if(i == k) break;
//    centroids[t] = centroids_pp[i][j];
//    ++j;
//  }
//  // -----------------------------------------------------------------------------------
//  iteration_num = 0;
//  valid = 1;
//  while (iteration_num < max_iter && valid == 1) {
//    for (i = 0; i < k * d; i++) {
//      old_centroids[i] = centroids[i];
//    }
//    for (i = 0; i < n; i++) {
//      int index = index_of_closest_cluster(data_points[i], centroids, k, d);
//      for (j = 0; j < d; j++) {
//        clusters[index * d + j] += data_points[i][j];
//      }
//      sizeof_clusters[index]++;
//    }
//    for (j = 0; j < k; j++) {
//      for (i = 0; i < d; i++) {
//        centroids[d * j + i] = clusters[d * j + i] / sizeof_clusters[j];
//      }
//    }
//    for (j = 0; j < k * d; j++) {
//      clusters[j] = 0;
//    }
//    for (j = 0; j < k; j++) {
//      sizeof_clusters[j] = 0;
//    }
//    valid = is_converged(centroids, old_centroids, k, d, epsilon);
//    iteration_num++;
//  }
//  free(clusters);
//  free(sizeof_clusters);
//  free(old_centroids);
//  free(vectors);
//  for (i = 0; i < n; i++) {
//    free(data_points[i]);
//  }
//  free(data_points);
  return 0;
}

/*-----------------CAPI------------------*/
static PyObject *fit(int k, int maxIter, int n, int d, double epsilon, PyObject *data_points.tolist(), PyObject *centroids.tolist()) {
    /* ~change the inputs from python to c:~ */
    double *currcentroids, **points;
    PyObject * line, *member, *PYresult;
    int i, j;
    PyObject *finalmergeList = data_points.tolist();
    PyObject *kcentroids = centroids.tolist();
    currcentroids = (double *) malloc(d * k * sizeof(double));
    points = (double **) malloc(n * sizeof(double *));
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            member = PyList_GetItem(kcentroids, i * d + j);
            currcentroids[i * d + j] = PyFloat_AsDouble(member);
        }
    }
    for (i = 0; i < n; i++) {
        double *point = malloc(d * sizeof(double));
        line = PyList_GetItem(finalmergeList, i);
        for (j = 0; j < d; j++) {
            member = PyList_GetItem(line, j);
            point[j] = PyFloat_AsDouble(member);
        }
        points[i] = point;
    }
    k_meansM(points, currcentroids, epsilon, k, d, n, maxIter);

    PYresult = PyList_New(k);
    for (i = 0; i < k; i++) {
        PyObject * PYcentroid = PyList_New(d);
        for (j = 0; j < d; j++) {
            PyObject * mem = PyFloat_FromDouble(currcentroids[i * d + j]);
            PyList_SET_ITEM(PYcentroid, j, mem);
        }
        PyList_SET_ITEM(PYresult, i, PYcentroid);
    }

    for (i = 0; i < n; i++) {
        free(points[i]);
    }
    free(points);
    free(currcentroids);
    return PYresult;
}

static PyObject *k_meansM(PyObject *self, PyObject *args) {
    int k, maxIter, n, d;
    double epsilon;
    PyObject * finalmergeList, *kcentroids;
    /*
     * In the C/Python API, a NULL value is never valid for a PyObject*, so it's used to signal that an error has happened.
     */
    if (!PyArg_ParseTuple(args, "iiiidOO", &k, &maxIter, &n, &d, &epsilon, &finalmergeList, &kcentroids)) {
        return NULL;
    }
    return fit(k, maxIter, n, d, epsilon, finalmergeList, kcentroids);
}





// static PyObject* fit(PyObject *self, PyObject *args) {
//     int k, max_iter, n, d;
//     double epsilon;
//     PyObject *data_points_from_python, *centroids_from_python;
//     double** data_points;
//     double** centroids_pp;
//     /*
//      * In the C/Python API, a NULL value is never valid for a PyObject*, so it's used to signal that an error has happened.
//      */
//     if (!PyArg_ParseTuple(args, "iiiidOO", &k, &max_iter, &n, &d, &epsilon,
//      &data_points_from_python, &centroids_from_python)) {
//         return NULL;
//     }
//     data_points = read_2d_array_from_python(n, d, data_points_from_python);
//     centroids_pp = read_2d_array_from_python(k, d, centroids_from_python);
//     if (run_kmeans(data_points, centroids_pp, n, d, k, max_iter, epsilon) != 0) {
//       free_mat(data_points, n);
//       free_mat(centroids_pp, k);
//       return NULL;
//     } else {
//       free_mat(data_points, n);
//       free_mat(centroids_pp, k);
//       return pass_2d_array_to_python(k, d, centroids_pp);
//     }
// }
// static double** read_2d_array_from_python(int rows, int cols, PyObject *py_list) {
//   int i, j;
//   PyObject *row_i, *data;
//   double **arr;
//   arr = calloc(rows, sizeof(double*));
//   for (i = 0; i < rows; ++i) {
//     arr[i] = calloc(cols, sizeof(double));
//     row_i = PyList_GetItem(py_list, i);
//     for (j = 0; j < cols; ++j) {
//       data = PyList_GetItem(row_i, j);
//       arr[i][j] = PyFloat_AsDouble(data);
//     }
//   }
//   return arr;
// }

// static PyObject* pass_2d_array_to_python(int rows, int cols, double **arr) {
//   int i, j;
//   PyObject *row_i, *arr_to_python, *data;
//   arr_to_python = PyList_New(rows);
//   for (i = 0; i < rows; ++i) {
//     row_i = PyList_New(cols);
//     for (j = 0; j < cols; ++j) {
//       data = PyFloat_FromDouble(arr[i][j]);
//       PyList_SET_ITEM(row_i, j, data);
//     }
//     PyList_SET_ITEM(arr_to_python, i, row_i);
//   }
//   return arr_to_python;
// }

/*
 * This array tells Python what methods this module has.
 */
static PyMethodDef capiMethods[] = {
        {"fit", /* The Python method name that will be used. */
         (PyCFunction) fit, /* the C function that implements the Python function and returns static PyObject*. */
         METH_VARARGS, /* States that the functions has arguments. */
         PyDoc_STR("fit() method") /* The docstring for the function. */
         },
        {NULL, NULL, 0, NULL} /* The last entry must be all NULL as shown to act as a sentinel.
                               * Python looks for this entry to know that all of the functions
                               * for the module have been defined. */
};

/*
 * This initiates the module using the above definitions.
 */
static struct PyModuleDef kmeansspModule = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp", /* name of module. */
        NULL, /* module documentation, may be NULL. */
        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        capiMethods /* the PyMethodDef array from before containing the methods of the extension. */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module's
 * initialization function. The initialization function must be named PyInit_name(), where
 * name is the name of the module and should match what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file.
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void) {
    PyObject * m;
    m = PyModule_Create(&kmeansspModule);
    if (!m) {
        return NULL;
    }
    return m;
}

