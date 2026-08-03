#ifndef _NFA_
#define _NFA_

#include "../../camera_model/vio_def.h"
#define TABSIZE 100000

//----------------------------------------------
// Fast arctan2 using a lookup table
//
#define MAX_LUT_SIZE 1024

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif /* !M_PI */

#define RELATIVE_ERROR_FACTOR 100.0

namespace dso {
// Lookup table (LUT) for NFA computation
class NFALUT {
 public:
  NFALUT(int size, number_t _prob, number_t _logNT);
  ~NFALUT();

  int* LUT;  // look up table
  int LUTSize;

  number_t prob;
  number_t logNT;

  bool checkValidationByNFA(int n, int k);
  static number_t myAtan2(number_t yy, number_t xx);

 private:
  number_t nfa(int n, int k);
  static number_t log_gamma_lanczos(number_t x);
  static number_t log_gamma_windschitl(number_t x);
  static number_t log_gamma(number_t x);
  static int number_t_equal(number_t a, number_t b);
};
}  // namespace dso

#endif