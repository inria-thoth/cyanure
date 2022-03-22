#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <stdexcept>

class NotImplementedException
    : public std::exception
{

public:
    // Construct with given error message:
    NotImplementedException(const char *error = "Functionality not yet implemented!")
    {
        errorMessage = error;
    }

    // Provided for compatibility with std::exception.
    const char *what() const noexcept
    {
        return errorMessage.c_str();
    }

private:
    std::string errorMessage;
};

class ValueError
    : public std::exception
{

public:
    // Construct with given error message:
    ValueError(const char *error = "The value is not valid for the parameter!")
    {
        errorMessage = error;
    }

    // Provided for compatibility with std::exception.
    const char *what() const noexcept
    {
        return errorMessage.c_str();
    }

private:
    std::string errorMessage;
};

class ConversionError
    : public std::exception
{

public:
    // Construct with given error message:
    ConversionError(const char *error = "An error has occured during the conversion between Python and C++.")
    {
        errorMessage = error;
    }

    // Provided for compatibility with std::exception.
    const char *what() const noexcept
    {
        return errorMessage.c_str();
    }

private:
    std::string errorMessage;
};

#endif