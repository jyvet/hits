/**
* HIP Transfer Streams (HIts): Application designed to launch intra-node
*                               transfer streams in an adjustable way.
* URL       https://github.com/jyvet/hits
* License   MIT
* Author    Jean-Yves VET <contact[at]jean-yves.vet>
* Copyright (c) 2023
******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <argp.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <pthread.h>
#include <numa.h>
#include <assert.h>

/* Expand macro values to string */
#define STR_VALUE(var)  #var
#define STR(var)        STR_VALUE(var)

#define N_SIZE_MAX      1073741824  /* 1GiB */
#define N_SIZE_DEFAULT  N_SIZE_MAX
#define N_ITER_DEFAULT  100
#define HITS_VERSION    "hits 1.1"
#define HITS_CONTACT    "https://github.com/jyvet/hits"

#define checkHip(ret) { assertHip((ret), __FILE__, __LINE__); }
inline void assertHip(hipError_t code, const char *file, int line)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"CheckHip: %s %s %d\n", hipGetErrorString(code), file, line);
      exit(code);
   }
}

typedef enum TransferType
{
    HTOD = 0,  /* Host memory to Device (GPU)  */
    DTOH,      /* Device (GPU) to Host memory  */
    DTOD,      /* Device (GPU) to Device (GPU) */
} TransferType_t;

const char * const ttype_str[] =
{
    "Host to Device",
    "Device to Host",
    "Device to Device",
};

typedef struct Transfer
{
    hipEvent_t      start;      /* Start event for timing purpose                */
    hipEvent_t      stop;       /* Stop event for timing purpose                 */
    int             device;     /* First (or single) device involved in transfer */
    int             device2;    /* Second device involved in the transfer        */
    float          *dest;       /* Source buffer (host or GPU memory)            */
    float          *src;        /* Destination buffer (host or GPU memory)       */
    hipStream_t     stream;     /* HIP stream dedicated to the transfer          */
    TransferType_t  type;       /* Type and direction of the transfer            */
    int             numa_node;  /* NUMA node locality                            */
    bool            is_started; /* True if at least one stream event submitted   */
    struct hipDeviceProp_t prop_device;
    struct hipDeviceProp_t prop_device2;
} Transfer_t;

enum Flags
{
    is_numa_aware = 1 << 0,
    is_pinned     = 1 << 1,
};

typedef struct Hits
{
    Transfer_t *transfer;      /* Array containing all transfers to launch     */
    int         n_transfers;   /* Amount of transfers                          */
    long        n_iter;        /* Amount of iterations for each transfer       */
    long        n_size;        /* Transfer size in bytes                       */
    int         alloc_flags;   /* Allocation flags (NUMA aware and pinned)     */
} Hits_t;

const char *argp_program_version = HITS_VERSION;
const char *argp_program_bug_address = HITS_CONTACT;

/* Program documentation */
static char doc[] = "This application is designed to launch intra-node transfer streams "
                    "in an adjustable way. It may trigger different types of transfers "
                    "concurrently. Each transfer is bound to a stream. Transfer "
                    "buffers in main memory are allocated (by default) on the proper "
                    "NUMA node. The application accepts the following arguments:";

/* A description of the arguments we accept (in addition to the options) */
static char args_doc[] = "--dtoh=<gpu_id> --htod=<gpu_id> --dtod=<dest_gpu_id,src_gpu_id>";

/* Options */
static struct argp_option options[] =
{
    {"dtoh",                  'd', "<id>",    0,  "Provide GPU id for Device to Host transfer."},
    {"htod",                  'h', "<id>",    0,  "Provide GPU id for Host to Device transfer."},
    {"dtod",                  'p', "<id,id>", 0,  "Provide comma-separated GPU ids to specify which "
                                                  "pair of GPUs to use for peer to peer transfer. "
                                                  "First id is the destination, second id is the source."},
    {"iter",                  'i', "<nb>",    0,  "Specify the amount of iterations. [default: "
                                                  STR(N_ITER_DEFAULT) "]"},
    {"disable-numa-affinity", 'n', 0,         0,  "Do not make the transfer buffers NUMA aware."},
    {"disable-pinned-memory", 'm', 0,         0,  "Use pageable allocations instead."},
    {"size",                  's', "<bytes>", 0,  "Specify the transfer size in bytes. [default: "
                                                  STR(N_SIZE_DEFAULT) "]"},
    {0}
};

/* Parse a single option */
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    Hits_t *hits = (Hits_t *)state->input;
    Transfer_t *transfer = &hits->transfer[hits->n_transfers];

    const char* token;
    char *endptr;

    switch (key)
    {
        case 'd':
            transfer->type = DTOH;
            transfer->device = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse the GPU id from the --dtoh argument. "
                                "Exit.\n");
                exit(1);
            }

            transfer->device2 = -1;
            hits->n_transfers++;
            break;
        case 'h':
            transfer->type = HTOD;
            transfer->device = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse the GPU id from th --htod argument. "
                                "Exit.\n");
                exit(1);
            }

            transfer->device2 = -1;
            hits->n_transfers++;
            break;
        case 'i':
            hits->n_iter = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || hits->n_iter < 0)
            {
                fprintf(stderr, "Error: cannot parse the amount of iterations from the "
                                "--iter argument. Exit.\n");
                exit(1);
            }
            break;
        case 'n':
            hits->alloc_flags = hits->alloc_flags & ~is_numa_aware;
            break;
        case 'm':
            hits->alloc_flags = hits->alloc_flags & ~is_pinned;
            break;
        case 'p':
            transfer->type = DTOD;

            /* Parse first GPU id */
            token = strtok(arg, ",");
            transfer->device = (token != NULL ) ? strtol(token, &endptr, 10) : -1;
            if (errno == EINVAL || errno == ERANGE || token == endptr || transfer->device < 0)
            {
                fprintf(stderr, "Error: cannot parse first GPU id from --dtod argument. "
                                "This argument only accepts a list of two ids separated "
                                "by a comma. Exit.\n");
                exit(1);
            }

            /* Parse second GPU id */
            token = strtok(NULL, ",");
            transfer->device2 = (token != NULL) ? strtol(token, &endptr, 10) : -1;
            if (errno == EINVAL || errno == ERANGE || token == endptr || transfer->device2 < 0)
            {
                fprintf(stderr, "Error: cannot parse second GPU id from --dtod argument. "
                                "This argument only accepts a list of two ids separated "
                                "by a comma. Exit.\n");
                exit(1);
            }

            /* Ensure there is no further ids */
            token = strtok(NULL, ",");
            if (token != endptr && token != NULL)
            {
                fprintf(stderr, "Error: --dtod argument only accepts a list of two GPU ids "
                                "separated by a comma. Exit.\n");
                exit(1);
            }

            hits->n_transfers++;
            break;
        case 's':
            hits->n_size = strtol(arg, &endptr, 10);
            if (errno == EINVAL || errno == ERANGE || hits->n_iter < 0)
            {
                fprintf(stderr, "Error: cannot parse the transfer size from the "
                                "--size argument. Exit.\n");
                exit(1);
            }

            if (hits->n_size > N_SIZE_MAX)
            {
                fprintf(stderr, "Error: maximum transfer size value is %d. Exit.", N_SIZE_MAX);
                exit(1);
            }
            break;
        case ARGP_KEY_END:
            if (hits->n_transfers == 0)
                argp_usage(state);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

/* Argp parser */
static struct argp argp = { options, parse_opt, args_doc, doc };

static void _transfer_init_common(Transfer_t *t)
{
    t->numa_node  = -1;
    t->is_started = false;

    checkHip( hipGetDeviceProperties(&t->prop_device, t->device) );
    if (t->device2 >= 0)
        checkHip( hipGetDeviceProperties(&t->prop_device2, t->device2) );

    checkHip( hipSetDevice(t->device) );

    checkHip( hipEventCreate(&t->start) );
    checkHip( hipEventCreate(&t->stop) );

    checkHip( hipStreamCreateWithFlags(&t->stream, hipStreamNonBlocking) );
}

/**
 * Set NUMA affinity based on GPU property.
 *
 * @param   t[in]  transfer structure
 */
void set_numa_affinity(Transfer_t *t)
{
    char numa_file[PATH_MAX];
    struct hipDeviceProp_t *prop = &t->prop_device;
    sprintf(numa_file, "/sys/class/pci_bus/%04x:%02x/device/numa_node",
                       prop->pciDomainID, prop->pciBusID);

    FILE* file = fopen(numa_file, "r");
    if (file == NULL)
        return;

    int ret = fscanf(file, "%d", &t->numa_node);
    fclose(file);

    if (ret == 1)
        numa_set_preferred(t->numa_node);
}

void dtoh_transfer_init(Transfer_t *t, const size_t n_bytes, const int alloc_flags)
{
    _transfer_init_common(t);

    if (alloc_flags & is_numa_aware)
        set_numa_affinity(t);

    if (alloc_flags & is_pinned)
    {
        checkHip( hipHostMalloc(((void **)&t->dest), n_bytes, hipHostMallocDefault | hipHostMallocNumaUser) );
    }
    else
        t->dest = (float *)malloc(n_bytes);

    assert(t->dest != NULL);

    checkHip( hipMalloc(((void **)&t->src), n_bytes) );
}

void htod_transfer_init(Transfer_t *t, const size_t n_bytes, const int alloc_flags)
{
    _transfer_init_common(t);

    if (alloc_flags & is_numa_aware)
        set_numa_affinity(t);

    if (alloc_flags & is_pinned)
    {
        checkHip( hipHostMalloc(((void **)&t->src), n_bytes, hipHostMallocDefault | hipHostMallocNumaUser) );
    }
    else
        t->src = (float *)malloc(n_bytes);

    assert(t->src != NULL);

    checkHip( hipMalloc(((void **)&t->dest), n_bytes) );
}

void dtod_transfer_init(Transfer_t *t, const size_t n_bytes)
{
    _transfer_init_common(t);

    /* Ensure peer-to-peer access is possible between the two GPUs */
    int is_access = 0;
    hipDeviceCanAccessPeer(&is_access, t->device, t->device2);
    if (!is_access)
    {
        fprintf(stderr, "Error: P2P cannot be enabled between devices %d and %d\n",
                t->device, t->device2);
        exit(1);
    }

    checkHip( hipSetDevice(t->device) );
    checkHip( hipMalloc((void **)&t->dest, n_bytes) );
    checkHip( hipDeviceEnablePeerAccess(t->device2, 0) );

    checkHip( hipSetDevice(t->device2) );
    checkHip( hipMalloc((void **)&t->src, n_bytes) );
}

/**
 * Initialize all transfers
 *
 * @param   hits[inout]  Main application structure
 */
void transfer_init(Hits_t *hits)
{
    /* Initialize all streams and buffers */
    for (int i = 0; i < hits->n_transfers; i++)
    {
        Transfer_t *t = &hits->transfer[i];

        switch(t->type)
        {
            case DTOH:
                dtoh_transfer_init(t, hits->n_size, hits->alloc_flags);
                break;
            case HTOD:
                htod_transfer_init(t, hits->n_size, hits->alloc_flags);
                break;
            case DTOD:
                dtod_transfer_init(t, hits->n_size);
                break;
        }
    }
}

/**
 * Initialize the application
 *
 * @param   argc[in]    Amount of arguments
 * @param   argv[in]    Array of arguments
 * @param   hits[out]   Main application structure
 */
void init(int argc, char *argv[], Hits_t *hits)
{
    hits->transfer = (Transfer_t *)malloc(sizeof(Transfer_t) * (argc - 1));
    if (hits->transfer == NULL)
    {
         fprintf(stderr,"Error: Cannot allocate main data structure. Exit.\n");
         exit(1);
    }

    /* Set defaults */
    hits->n_transfers   = 0;
    hits->n_iter        = N_ITER_DEFAULT;
    hits->n_size        = N_SIZE_DEFAULT;
    hits->alloc_flags   = is_numa_aware | is_pinned;

    argp_parse(&argp, argc, argv, 0, 0, hits);

    transfer_init(hits);
}

/**
 * Cleanup the application
 *
 * @param   hits[inout]  Main application structure
 */
void fini(Hits_t *hits)
{
    /* Free host buffers */
    for (int i = 0; i < hits->n_transfers; i++)
    {
        Transfer_t *t = &hits->transfer[i];

        switch(t->type)
        {
            case DTOH:
                if (hits->alloc_flags & is_pinned)
                {
                    checkHip( hipHostFree(t->dest) );
                }
                else
                    free(t->dest);
                break;
            case HTOD:
                if (hits->alloc_flags & is_pinned)
                {
                    checkHip( hipHostFree(t->src) );
                }
                else
                    free(t->src);
                break;
            case DTOD:
                break;
        }
    }

    free(hits->transfer);
}

/**
 * Launch a direct transfer stream (Host to Device or Device to Host)
 *
 * @param   t[inout]     Transfe data
 * @param   n_bytes[in]  Transfer size
 * @param   n_iter[in]   Iterations
 */
void direct_transfer(Transfer_t *t, const size_t n_bytes, const bool is_last_iter)
{
    checkHip( hipSetDevice(t->device) );

    if (!t->is_started)
    {
        printf("Launching %s transfers with Device %d (0x%.2x)",
               ttype_str[t->type], t->device, t->prop_device.pciBusID);

        if (t->numa_node >= 0)
            printf(" - Host buffer allocated on NUMA node %d", t->numa_node);

        printf("\n");

        checkHip( hipEventRecord(t->start, t->stream) );
        t->is_started = true;
    }

    checkHip( hipMemcpyAsync(t->dest, t->src, n_bytes, (t->type == DTOH) ?
                               hipMemcpyDeviceToHost : hipMemcpyHostToDevice, t->stream) );

    if (is_last_iter)
        checkHip( hipEventRecord(t->stop, t->stream) );
}

/**
 * Launch a peer-to-peer transfer stream
 *
 * @param   t[inout]     Transfe data
 * @param   n_bytes[in]  Transfer size
 * @param   n_iter[in]   Iterations
 */
void dtod_transfer(Transfer_t *t, const size_t n_bytes, const bool is_last_iter)
{
    checkHip( hipSetDevice(t->device) );

    if (!t->is_started)
    {
        printf("Launching P2P PCIe transfers from Device %d (0x%.2x) to Device %d (0x%.2x)\n",
               t->device2, t->prop_device2.pciBusID, t->device, t->prop_device.pciBusID);

        checkHip( hipEventRecord(t->start, t->stream) );
        t->is_started = true;
    }

    checkHip( hipMemcpyPeerAsync(t->dest, t->device, t->src, t->device2, n_bytes, t->stream) );

    if (is_last_iter)
        checkHip( hipEventRecord(t->stop, t->stream) );
}

/**
 * Display a dot every second as Heartbeat. Stop when transfers are completed.
 *
 * @param   arg[in]  Pointer to transfer state
 */
void* heart_beat(void *arg)
{
    bool *is_transfering = (bool*)arg;
    setbuf(stdout, NULL);

    while (*is_transfering)
    {
        sleep(1);
        printf(".");
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    Hits_t hits;

    init(argc, argv, &hits);
    const int n_transfers = hits.n_transfers;
    const size_t n_iter = hits.n_iter;
    const size_t n_bytes = hits.n_size;
    const float n_gbytes = (float)n_bytes / 1E9;
    bool is_transfering = true;
    pthread_t thread;

    /* Starting heartbeat thread */
    pthread_create(&thread, NULL, &heart_beat, &is_transfering);

    /* Start all transfers at the same time */
    for (size_t i = 0; i < n_iter; i++)
    {
        const bool is_last = (i == n_iter - 1);
        for (int j = 0; j < n_transfers; j++)
        {
            Transfer_t *t = &hits.transfer[j];
            (t->type == DTOD) ? dtod_transfer(t, n_bytes, is_last) : direct_transfer(t, n_bytes, is_last);
        }
    }

    /* Synchronize the GPU from each transfer */
    for (int i = 0; i < n_transfers; i++)
    {
        Transfer_t *t = &hits.transfer[i];
        checkHip( hipSetDevice(t->device) );
        checkHip( hipDeviceSynchronize() );
    }

    is_transfering = false;
    printf("\nCompleted.\n");

    /* Print bandwidth results */
    for (int i = 0; i < n_transfers; i++)
    {
        float dt_msec, dt_sec;
        Transfer_t *t = &hits.transfer[i];
        checkHip( hipSetDevice(t->device) );
        checkHip( hipEventElapsedTime(&dt_msec, t->start, t->stop) );
        dt_sec = dt_msec / 1E3;

        if (t->type == DTOD)
            printf("Transfer %d - P2P transfers from Device %d (0x%.2x) to Device %d (0x%.2x):"
                   " %.3f GB/s  (%.2f seconds)\n", i, t->device2, t->prop_device2.pciBusID,
                   t->device, t->prop_device.pciBusID, n_gbytes / dt_sec * n_iter, dt_sec);
        else
            printf("Transfer %d - Direct transfers (%s) with Device %d (0x%.2x): "
                   "%.3f GB/s  (%.2f seconds)\n", i, ttype_str[t->type],
                   t->device, t->prop_device.pciBusID, n_gbytes / dt_sec * n_iter, dt_sec);
    }

    fini(&hits);

    return 0;
}

