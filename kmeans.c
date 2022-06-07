#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
/******************************************************************************

@author: mohammad daghash
@id: 314811290
@author: ram elgov
@id: 206867517

Implementation of the KMEANS algorithm from HW1.
Minor changes was added to support the interaction with the C API
and the input from python. The use of KMEANS++ initial centroids was added too.

*******************************************************************************/

static void free_2d_arr(double** m, int rows) {
/*
function to free memory allocated for 2d array.
*/
  int i;
  for (i = 0; i < rows; ++i) {
    free(m[i]);
  }
  free(m);
}
static int is_converged(double *centroids, double *old_centroids, int K, int d, double epsilon) {
/*
checks if the convergence criteria has been reached.
*/
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
/*
calculating the index of the closest cluster to the given data point.
*/
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
static int run(double** data_points, double** centroids_pp, int n, int d, int k, int max_iter, double epsilon) {
/*
the main clustering algorithm using kmeans.
same implementation from HW1 except using kmeans++ and data parsing implemented in python.
*/
  int iteration_num, valid, i, j, t;
  double *vectors;
  double *centroids;
  double *old_centroids;
  double *clusters;
  int *sizeof_clusters;
  centroids = calloc(k * d, sizeof(double));
  vectors = calloc(d * n, sizeof(double));
  old_centroids = calloc(k * d, sizeof(double));
  clusters = calloc(k * d, sizeof(double));
  sizeof_clusters = calloc(k, sizeof(int));

  // kmeans++ centroids initialization ------------------------------------------------------
  t = 0;
  i = 0;
  j = 0;
  while (t < d * k) {
    if (j == d) {
      j = 0;
      ++i;
    }
    if(i == k) break;
    centroids[t] = centroids_pp[i][j];
    ++j;
    ++t;
  }
  // -----------------------------------------------------------------------------------
  iteration_num = 0;
  valid = 1;
  while (iteration_num < max_iter && valid == 1) {
    for (i = 0; i < k * d; i++) {
      old_centroids[i] = centroids[i];
    }
    for (i = 0; i < n; i++) {
      int index = index_of_closest_cluster(data_points[i], centroids, k, d);
      for (j = 0; j < d; j++) {
        clusters[index * d + j] += data_points[i][j];
      }
      sizeof_clusters[index]++;
    }
    for (j = 0; j < k; j++) {
      for (i = 0; i < d; i++) {
        centroids[d * j + i] = clusters[d * j + i] / sizeof_clusters[j];
      }
    }
    for (j = 0; j < k * d; j++) {
      clusters[j] = 0;
    }
    for (j = 0; j < k; j++) {
      sizeof_clusters[j] = 0;
    }
    valid = is_converged(centroids, old_centroids, k, d, epsilon);
    iteration_num++;
  }
  t = 0;
  i = 0;
  j = 0;
  while (t < d * k) {
    if (j == d) {
      j = 0;
      ++i;
    }
    if(i == k) break;
    centroids_pp[i][j] = centroids[t];
    ++j;
    ++t;
  }
  free(clusters);
  free(sizeof_clusters);
  free(centroids);
  free(old_centroids);
  free(vectors);
  return 0;
}
/*
C API code
*/
static double** get_from_python(int num_of_elements, int dim, PyObject *python_list){
/*
parse python list input into 2d array.
*/
    int i, j;
    double **matrix;
    PyObject *temp_list, *element;
    matrix = calloc(num_of_elements, sizeof(double*));
    for (i = 0; i < num_of_elements; i++){
        matrix[i] = calloc(dim, sizeof(double));
        temp_list = PyList_GetItem(python_list, i);
        for (j = 0; j < dim; j++){
            element = PyList_GetItem(temp_list, j);
            matrix[i][j] = PyFloat_AsDouble(element);
        }
    }
    return matrix;
}
static PyObject* send_to_python(double** centroids, int K, int dim){
/*
send the final centroids to python as a list object.
*/
    int i, j;
    PyObject* outer_list;
    PyObject* inner_list;
    PyObject* element;
    outer_list = PyList_New(K);
    for (i = 0; i < K; i++){
        inner_list = PyList_New(dim);
        for (j = 0; j < dim; j++){
            element = PyFloat_FromDouble(centroids[i][j]);
            PyList_SET_ITEM(inner_list, j, element);
        }
        PyList_SET_ITEM(outer_list, i, inner_list);
    }
    return outer_list;
}
static PyObject* fit(PyObject *self, PyObject *args) {
/*
the algorithm's fit() function. calls run() and return the output back to python.
*/
    PyObject *output, *data_points_list, *centroid_list;
    int N, K, max_iter, dim;
    double **centroids, **data_points;
    double epsilon;
    if (!PyArg_ParseTuple(args, "iiiidOO", &N, &K, &max_iter, &dim, &epsilon,
    &centroid_list, &data_points_list)){
        return NULL;
    }
    data_points = get_from_python(N, dim, data_points_list);
    centroids = get_from_python(K, dim, centroid_list);
    if (run(data_points, centroids, N, dim, K, max_iter, epsilon))
    {
        free_2d_arr(data_points, N);
        free_2d_arr(centroids, K);
        return NULL;
    }
    else {
        output = send_to_python(centroids, K, dim);
        free_2d_arr(data_points, N);
        free_2d_arr(centroids, K);
        return output;
    }
}
/*
the functions below are simple C/API configurators.
*/
static PyMethodDef capiMethods[] = {
    {"fit", (PyCFunction) fit, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    capiMethods
};
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}
