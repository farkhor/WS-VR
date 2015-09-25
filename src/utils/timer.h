#ifndef TIMER_H_
#define TIMER_H_

#include <stdlib.h>	// for timing
#include <sys/time.h>	// for timing

namespace timer {
	timeval StartingTime;
	void setTime(){
		gettimeofday( &StartingTime, NULL );
	}
	double getTime(){
		timeval PausingTime, ElapsedTime;
		gettimeofday( &PausingTime, NULL );
		timersub(&PausingTime, &StartingTime, &ElapsedTime);
		return ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;	// Returning in milliseconds.
	}
}

#endif /* TIMER_H_ */
