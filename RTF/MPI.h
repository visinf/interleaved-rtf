#ifndef _H_RTF_MPI_H_
#define _H_RTF_MPI_H_

#include "Types.h"

#define MSMPI_NO_DEPRECATE_20 1
#ifdef USE_MPI
#include <boost/mpi.hpp>

namespace MPI
{
    boost::mpi::communicator& Communicator()
    {
        static boost::mpi::communicator comm;
        return comm;
    }

    boost::mpi::environment& Environment(int argc = 0, char** argv = NULL)
    {
        static boost::mpi::environment env(argc, argv);
        return env;
    }
}

#endif // USE_MPI

#endif // _H_RTF_MPI_H_