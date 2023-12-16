#include<stdio.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<unistd.h>
#include<pthread.h>

#define PORT 10000
#define BUFFER_SIZE 256
#define	MAX_CLIENTS 20

void* thread_job(void*);
pthread_mutex_t lock;
int counter=0;


int main(int argc, char **argv){

	int iResult;

	int listener=socket(AF_INET, SOCK_STREAM, 0);
	if (listener == -1) {
		printf("Error creating socket\n");
		return 1;
	}

	sockaddr_in  listenerConf;
	listenerConf.sin_port=htons(PORT);
	listenerConf.sin_family=AF_INET;
	listenerConf.sin_addr.s_addr=INADDR_ANY;

	iResult = bind( listener, (sockaddr *)&listenerConf, sizeof(listenerConf));
	if (iResult == -1) {
		printf("Bind failed\n");
		close(listener);
		return 1;
	}

	if ( listen( listener, 5 ) == -1 ) {
		printf( "Listen failed\n");
		close(listener);
		return 1;
	}

	int clientSock;
	char buff[BUFFER_SIZE];
	
	pthread_mutex_init(&lock,NULL);

	while (1)
	{
		clientSock = accept(listener,NULL,NULL);
		if (clientSock == -1) {
			printf("Accept failed\n");
			close(listener);
			return 1;
		}
		
		pthread_mutex_lock(&lock);
		if(counter >= MAX_CLIENTS ){
			pthread_mutex_unlock(&lock);
			close(clientSock);
			continue;
		}
		pthread_mutex_unlock(&lock);

		pthread_t t1;
		pthread_create(&t1,NULL,thread_job,(void *) &clientSock);
	}

	close(listener);

	return 0;
}

void* thread_job(void* clientS){

	pthread_mutex_lock(&lock);
	counter++;
	int num=counter;
	pthread_mutex_unlock(&lock);

	int iResult=0;
	char buff[BUFFER_SIZE];
	int clientSock=*((int *) clientS);
	do{
		iResult = recv(clientSock, buff, BUFFER_SIZE, 0);
		if (iResult > 0) {
			printf("Client num %d,R : %dB\n",num, iResult);

			iResult = send(clientSock, buff, iResult, 0 );
			if (iResult == -1) {
				printf("send failed!\n");
				close(clientSock);
				break;
			}
			printf("Client num %d,S : %dB\n", num,iResult);
		}
		else if (iResult == 0){
			pthread_mutex_lock(&lock);
			printf("Connection closing client %d of %d...\n",num,counter);
			pthread_mutex_unlock(&lock);
		}else{
			printf("recv failed!\n");
			close(clientSock);
			break;
		}

	} while (iResult > 0);
	close(clientSock);
	
	pthread_mutex_lock(&lock);
	counter--;
	pthread_mutex_unlock(&lock);
	
	return NULL;
}
