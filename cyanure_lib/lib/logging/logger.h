#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include <iostream>
#include <sstream>
#include <string>

/* consider adding boost thread id since we'll want to know whose writting and
 * won't want to repeat it for every single call */

/* consider adding policy class to allow users to redirect logging to specific
 * files via the command line
 */

enum loglevel_e
    {logERROR, logWARNING, logINFO, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};

static const char *loglevel_e_str[] =
        { "Error", "Warning", "Info", "Debug", "Debug1", "Debug2", "Debug3", "Debug4" };

class logIt
{
public:
    logIt(loglevel_e _loglevel = logERROR) {

        if (_loglevel == logWARNING)
            _buffer << "\033[33m" <<getStringForEnum(_loglevel) << " :" 
            << std::string(_loglevel > logDEBUG ? (_loglevel - logDEBUG) * 4 
                : 1, ' ') ;
        else if (_loglevel == logERROR)
            _buffer << "\033[31m" <<  getStringForEnum(_loglevel) << " :" 
                << std::string(_loglevel > logDEBUG ? (_loglevel - logDEBUG) * 4 
                    : 1, ' ') ;
        else
             _buffer <<  getStringForEnum(_loglevel) << " :" 
                << std::string(_loglevel > logDEBUG ? (_loglevel - logDEBUG) * 4 
                    : 1, ' ') ;
        
    }

    template <typename T>
    logIt & operator<<(T const & value)
    {
        _buffer << value;
        return *this;
    }

    ~logIt()
    {
        _buffer << "\033[0m" << std::endl;
        // This is atomic according to the POSIX standard
        // http://www.gnu.org/s/libc/manual/html_node/Streams-and-Threads.html
        std::cerr << _buffer.str();
    }

private:
    std::ostringstream _buffer;

    std::string getStringForEnum( int enum_val )
    {
        std::string tmp(loglevel_e_str[enum_val]);
        return tmp;
    }
};

extern loglevel_e loglevel;

#define logging(level) \
if (level > loglevel) ; \
else logIt(level)

#endif
