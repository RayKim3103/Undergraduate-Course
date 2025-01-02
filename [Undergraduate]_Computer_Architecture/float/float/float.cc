#include "float.h"

using namespace std;

// Class constructor
float64_t::float64_t(void) : data(0) { /* Nothing to do */ }

// Class constructor
float64_t::float64_t(const double &v) : data(*((uint64_t*)&v)) { /* Nothing to do */ }

// Class constructor (private)
float64_t::float64_t(const uint64_t d) : data(d) { /* Nothing to do */ }
   
// Class copy constructor
float64_t::float64_t(const float64_t &f) : data(f.data) { /* Nothing to do */ }

// Class destructor
float64_t::~float64_t(void) { /* Nothing to do */ }

// cout << operator
ostream& operator<<(ostream &os, const float64_t &f) { os << *((double*)&f.data); return os; }

// Assignment = operator
float64_t& float64_t::operator=(const double &v) { data = *((uint64_t*)&v); return *this; }

// Assignment = operator
float64_t& float64_t::operator=(const float64_t &f) { data = f.data; return *this; }

// Unary - operator
float64_t float64_t::operator-(void) { return float64_t(data ^ (uint64_t(1) << (exp_bits + frac_bits))); }





/*******************************************************************
 * EEE3530 Assignment #3: Floating-point numbers                   *
 * Implement the double-precision floating-point add and subtract  *
 * operators below. The functions should perform bit-level         *
 * operations to produce the results.                              *
 *******************************************************************/

// Add + operator
float64_t float64_t::operator+(const float64_t &y) {
    /***************************************************************
     * EEE3530 Assignment #3                                       *
     * Implement the double-precision floating-point add function. *
     ***************************************************************/

    // An example to extract the sign, exponent, and fraction of x (*this).
    // bool x_sign    = data >> (exp_bits + frac_bits);
    // int64_t x_exp  = (data & exp_mask) >> frac_bits;
    // int64_t x_frac = data & frac_mask;
    
    // An example to extract the sign, exponent, and fraction of y (arg).
    // bool y_sign    = y.data >> (exp_bits + frac_bits);
    // int64_t y_exp  = (y.data & exp_mask) >> frac_bits;
    // int64_t y_frac = y.data & frac_mask;

    // Put the calculated sign, exponent, and fraction into r.data.
    
    // sign, exp, frac of r
    float64_t r;
    uint64_t r_sign = 0;
    uint64_t r_exp = 0;
    uint64_t r_frac = 0;
    r = 0;

    // sign, exp, frac of x (*this)
    bool x_sign = data >> (exp_bits + frac_bits);
    uint64_t x_exp  = (data & exp_mask) >> frac_bits;
    uint64_t x_frac = data & frac_mask;

    // sign, exp, frac of y (arg)
    bool y_sign = y.data >> (exp_bits + frac_bits);
    uint64_t y_exp  = (y.data & exp_mask) >> frac_bits;
    uint64_t y_frac = y.data & frac_mask;

    // If one of x, y is nan than result is nan
    uint64_t result_nan = 0;

    // temporary value for sign, exp, frac of smaller number
    // temporary value is needed to shift frac bits of smaller number during addition
    bool small_sign;                                
    uint64_t small_exp;
    uint64_t small_frac;

    // temporary value for sign, exp, frac of big number
    bool big_sign;
    uint64_t big_exp;
    uint64_t big_frac;

    /********************* case 1. x or y is nan *********************/

    // x is nan, result is nan
    if (x_exp == exp_max && x_frac != 0)
    {
        r_sign = 0;
        r_exp = exp_max;
        r_frac = 1;      // non-zero value

        result_nan = 1;
    }
    
    // y is nan, result is nan
    else if (y_exp == exp_max && y_frac != 0)
    {
        r_sign = 0;
        r_exp = exp_max;
        r_frac = 1;      // non-zero value

        result_nan = 1;
    }

    /********************* case 2. x or y is infinite *********************/

    // x is infinte, result is infinite
    else if (x_exp == exp_max)
    {
            r_sign = x_sign;
            r_exp = exp_max;
            r_frac = 0;
        
    }

    // y is infinte, result is infinite
    else if (y_exp == exp_max)
    {
        r_sign = y_sign;
        r_exp = exp_max;
        r_frac = 0;

    }

    /********************* case 3. x or y is 0 *********************/

    // x = 0
    else if ((x_exp == 0) && (x_frac == 0)) 
    { 
        // r = y
        r_sign = y_sign;
        r_exp = y_exp;
        r_frac = y_frac;
    }

    // y = 0
    else if ((y_exp == 0) && (y_frac == 0)) 
    { 
        // r = x
        r_sign = x_sign;
        r_exp = x_exp;
        r_frac = x_frac;
    }

    /********************* case 4. Denormalized number *********************/

    // Both x, y is Denormalized number
    else if ((x_exp == 0) && (y_exp == 0))
    {
        // x is bigger than y
        if (x_frac > y_frac)
        {
            big_sign = x_sign; big_exp = x_exp; big_frac = x_frac;        // copy big number sign, exp, frac to temporary
            small_sign = y_sign; small_exp = y_exp; small_frac = y_frac;  // copy small number sign, exp, frac to temporary
            
        }
        // y is bigger or equal than x
        else
        {
            big_sign = y_sign; big_exp = y_exp; big_frac = y_frac;        // copy big number sign, exp, frac to temporary
            small_sign = x_sign; small_exp = x_exp; small_frac = x_frac;  // copy small number sign, exp, frac to temporary
        }

        r_sign = big_sign;  // sign of x + y follows the bigger value
        r_exp = big_exp;    // exp of x+y follows the bigger value

        // (cf) As, x & y has same exponent no shift operation

        // x, y has same sign
        if (big_sign == small_sign)
        {
            r_frac = big_frac + small_frac; // add fraction bits
            
            // if r_frac is bigger than frac_mask, result becomes normalized number
            // Thus, we don't shift but, do subtract operation : r_frac - (frac_mask+1), also r_exp = r_exp + 1
            if (r_frac > frac_mask)
            {
                r_exp += 1;
                r_frac -= (frac_mask+1);
            }
        }
        
        // x, y has different sign
        else
        {
            r_frac = big_frac - small_frac; // subtract fraction bits
        }

    }
    
    // x is denormalized or y is denormalized
    else if ((x_exp == 0) || (y_exp == 0))
    {
        // Denormalized number is smaller than Floating-point number
        // x is bigger than y
        if (y_exp == 0)
        {
            big_sign = x_sign; big_exp = x_exp; big_frac = x_frac;        // copy big number sign, exp, frac to temporary
            small_sign = y_sign; small_exp = y_exp; small_frac = y_frac;  // copy small number sign, exp, frac to temporary

        }
        // y is bigger or equal than x
        else
        {
            big_sign = y_sign; big_exp = y_exp; big_frac = y_frac;        // copy big number sign, exp, frac to temporary
            small_sign = x_sign; small_exp = x_exp; small_frac = x_frac;  // copy small number sign, exp, frac to temporary
        }
        r_sign = big_sign;  // sign of x + y follows the bigger value
        r_exp = big_exp;    // exp of x + y follows the bigger value
        small_frac = small_frac >> (big_exp - small_exp - 1); // shift smaller value fraction
                                                              // Reason of -1 : Denormalized number exp = -exp_bias + 1

        // x, y has same sign
        if (big_sign == small_sign)
        {
            r_frac = big_frac + small_frac; // add fraction bits

            // if r_frac is bigger than frac_mask, r_frac should be shift, and add 1 to r_exp
            // normalize
            if (r_frac > frac_mask)         
            {                               
                r_exp += 1;                 
                r_frac = (r_frac) >> 1;
            }
        }

        // x, y has different sign
        else
        {
            r_frac = (frac_mask+1) + big_frac - small_frac; // subtract fraction bits
                                                            // As, floating point number fraction is represented as 1 + a + b + ...; (frac_mask+1) considered 
            
            // As, x, y has different sign, x + y can be normalize number or denormalize number
            // when r_exp = 0 or 1, for both number and r_frac * 2 <= (frac_mask+1), the result is denormalize number
            if (r_frac < frac_mask)                         
            {
                // if x + y becomes normalize number
                if ((r_frac << 1) > (frac_mask + 1))
                {
                    // normalize
                    r_exp -= 1;
                    r_frac = r_frac << 1;
                    r_frac -= (frac_mask + 1);
                }
                // else, x + y becomes denormalize number
                else
                {
                    // don't have to normalize
                    r_exp = 0; // r_exp = 0 for denormalize number
                }
            }
            else
            {
                r_frac -= (frac_mask + 1);
            }
        }
    }

    /********************* case 5. Floating point number *********************/

    // x & y has same exp
    else if (x_exp == y_exp)
    {
        // x is bigger than y
        if (x_frac > y_frac)
        {
            big_sign = x_sign; big_exp = x_exp; big_frac = x_frac;        // copy big number sign, exp, frac to temporary
            small_sign = y_sign; small_exp = y_exp; small_frac = y_frac;  // copy small number sign, exp, frac to temporary

        }
        // y is bigger or equal than x
        else
        {
            big_sign = y_sign; big_exp = y_exp; big_frac = y_frac;        // copy big number sign, exp, frac to temporary
            small_sign = x_sign; small_exp = x_exp; small_frac = x_frac;  // copy small number sign, exp, frac to temporary
        }

        r_sign = big_sign;  // sign of x + y follows the bigger value
        r_exp = big_exp;    // exp of x+y follows the bigger value

        // (cf) As, x & y has same exponent no shift operation

        // x, y has same sign
        if (big_sign == small_sign)
        {
            r_frac = big_frac + small_frac; // add fraction bits
            r_frac = r_frac >> 1;            // normalize
                                             // (1 + a + b + ...) + (1 + c + d + ...) = (2 + e + f + ...), it should be divided by 2
            r_exp += 1;
        }

        // x, y has different sign
        else
        {
            r_frac = big_frac - small_frac; // subtract fraction bits
            if (r_frac == 0) // result = 0
            {
                r_sign = 0;
                r_exp = 0;
            }
            // normalize
            else
            {
                while (r_frac < frac_mask)      
                // (1 + a + b + ...) - (1 + c + d + ...) = (0 + e + f + ...), it should be mulitplied by 2
                // until, (0 + e + f + ...) becomes (1 + g + h + ...)
                {                               

                    r_exp -= 1;
                    r_frac = r_frac << 1;
                }
                r_frac -= (frac_mask + 1);
            }
            
        }
    }

    // x & y has different exp
    else
    {
        // x is bigger than y
        if (x_exp > y_exp)
        {
            big_sign = x_sign; big_exp = x_exp; big_frac = x_frac;        // copy big number sign, exp, frac to temporary
            small_sign = y_sign; small_exp = y_exp; small_frac = y_frac;  // copy small number sign, exp, frac to temporary
        }
        // y is bigger or equal than x
        else
        {
            big_sign = y_sign; big_exp = y_exp; big_frac = y_frac;        // copy big number sign, exp, frac to temporary
            small_sign = x_sign; small_exp = x_exp; small_frac = x_frac;  // copy small number sign, exp, frac to temporary
        }
        r_sign = big_sign;  // sign of x + y follows the bigger value
        r_exp = big_exp;    // exp of x + y follows the bigger value
        small_frac += (frac_mask + 1);
        small_frac = small_frac >> (big_exp - small_exp); // shift smaller value fraction
        
        // x, y has same sign
        if (big_sign == small_sign)
        {
            r_frac = big_frac + small_frac; // add fraction bits
            // normalize
            if (r_frac > frac_mask)
            {
                r_frac = (r_frac) >> 1;
                r_exp += 1;
            }
            
        }

        // x, y has different sign
        else
        {
            r_frac = (frac_mask + 1) + big_frac - small_frac; // subtract fraction bits
            // normalize
            while (r_frac < frac_mask)      // (0 + e + f + ...), it should be mulitplied by 2
            {                               // until, (0 + e + f + ...) becomes (1 + g + h + ...)
                r_exp -= 1;
                r_frac = r_frac << 1;
            }
            r_frac -= (frac_mask + 1);
        }
    }

    /********************* case 6. Result bigger than max number *********************/

    // In case of #test 10, the result should be infinite, not nan
    if ((r_exp >= exp_max) && (result_nan == 0))
    {
        r_exp = exp_max;
        r_frac = 0;
    }

    // put back the sign, exp, frac to r
    r.data |= r_sign << (exp_bits + frac_bits);
    r.data |= r_exp << frac_bits;
    r.data |= r_frac;

    return r;
}

// Subtract - operator
float64_t float64_t::operator-(const float64_t &y) {
    /***************************************************************
     * EEE3530 Assignment #3                                       *
     * Implement the double-precision floating-point sub function. *
     ***************************************************************/

    // Put the calculated sign, exponent, and fraction into r.data.
    // Subtract can be done by just reversing the sign of y and performing addition
    float64_t r;
    float64_t reverse_y;

    // sign, exp, frac of y (arg)
    uint64_t y_sign = y.data >> (exp_bits + frac_bits);
    uint64_t y_exp = (y.data & exp_mask) >> frac_bits;
    uint64_t y_frac = y.data & frac_mask;

    // reversing the sign of y (arg)
    if (y_sign == 1)
    {
	y_sign = 0;
    }
    else
    {
	y_sign = 1;
    }

    // put back the reversed sign, exp, frac to reverse_y
    reverse_y.data |= y_sign << (exp_bits + frac_bits);
    reverse_y.data |= y_exp << frac_bits;
    reverse_y.data |= y_frac;

    // performing addition
    r = *this + reverse_y;

    return r;
}

