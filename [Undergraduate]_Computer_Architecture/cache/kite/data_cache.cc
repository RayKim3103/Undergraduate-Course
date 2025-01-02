#include <cstdlib>
#include <iostream>
#include "data_cache.h"

// #define DEBUG

using namespace std;

data_cache_t::data_cache_t(uint64_t *m_ticks, uint64_t m_cache_size,
                           uint64_t m_block_size, uint64_t m_ways,
                           uint64_t m_victim_size) :
    memory(0),
    ticks(m_ticks),
    blocks(0),
    victim_cache(0),
    cache_size(m_cache_size),
    block_size(m_block_size),
    num_sets(0),
    num_ways(m_ways),
    block_offset(0),
    set_offset(0),
    block_mask(0),
    set_mask(0),
    num_accesses(0),
    num_misses(0),
    num_loads(0),
    num_stores(0),
    missed_inst(0) {
    // Calculate the block offset.
    uint64_t val = block_size;
    while(!(val & 0b1)) {
        val = val >> 1; block_offset++;
        block_mask = (block_mask << 1) | 0b1;
    }
    // Check if the block size is a multiple of doubleword.
    if((block_size & 0b111) || (val != 1)) {
        cerr << "Error: cache block size must be a multiple of doubleword" << endl;
        exit(1);
    }

    // Check if the number of ways is a power of two.
    val = num_ways;
    while(!(val & 0b1)) { val = val >> 1; }
    if(val != 1) {
        cerr << "Error: number of ways must be a power of two" << endl;
        exit(1);
    }

    // Calculate the number of sets.
    num_sets = cache_size / block_size / num_ways;
    // Calculate the set offset and mask.
    val = num_sets;
    while(!(val & 0b1)) {
        val = val >> 1; set_offset++;
        set_mask = (set_mask << 1) | 0b1;
    }
    set_offset += block_offset;
    set_mask = set_mask << block_offset;
    // Check if the number of sets is a power of two.
    if(val != 1) {
        cerr << "Error: number of sets must be a power of two" << endl;
        exit(1);
    }
    
    // Allocate cache blocks.
    blocks = new block_t*[num_sets]();
    for(uint64_t i = 0; i < num_sets; i++) { blocks[i] = new block_t[num_ways](); }

    // Create a victim cache.
    victim_cache = new victim_cache_t(m_victim_size);
}

data_cache_t::~data_cache_t() {
    // Deallocate the cache blocks.
    for(uint64_t i = 0; i < num_sets; i++) { delete [] blocks[i]; }
    delete [] blocks;

    // Destruct the victim cache.
    delete victim_cache;
}

// Connect to the lower-level memory.
void data_cache_t::connect(data_memory_t *m_memory) { memory = m_memory; }

// Is cache free?
bool data_cache_t::is_free() const { return !missed_inst; }

/************************************************************************
* Read Operation (ld instruction)
*  Case1: Data Cache Hit (when, valid & tag is matched)
*	1) update block access time
*	2) read rd value
*	3) number of load & accesses increase by 1
*  Case2: Data Cache Miss, Victim Cache Hit
*  	1) search victim cache components
*	2) when matched, 
*	   - update the Evicted block TAG and insert to Victim Cache
*	   - restore the TAG of victim block and insert to Data Cache
*	   - replay read
*	3) when miss-matched, go to Case3
*  Case3: Data Cache Miss, Victim Cache Miss
*	1) missed instruction = m_inst
*	2) go to memory and find the proper block
*	3) number of misses increase by 1
**************************************************************************/    

// Read data from cache.
void data_cache_t::read(inst_t *m_inst) {
    // Check the memory address alignment.
    uint64_t addr = m_inst->memory_addr;
    if(addr & 0b111) {
        cerr << "Error: invalid alignment of memory address " << addr << endl;
        exit(1);
    }

    // Calculate the set index and tag.
    uint64_t set_index = (addr & set_mask) >> block_offset;
    uint64_t tag = addr >> set_offset;
    
    // Check direct-mapped cache entry.
    block_t *block = &blocks[set_index][0];

    // Data Cache hit
    if(block->valid && block->tag == tag) 
    { 
        // Update the last access time.
        block->last_access = *ticks;
        // Read a doubleword in the block.
        m_inst->rd_val = *(block->data + ((addr & block_mask) >> 3));
#ifdef DATA_FWD
        m_inst->rd_ready = true;
#endif
	// increment accesses & loads
        num_accesses++;
        num_loads++;
//#ifdef DEBUG
//	cout << *ticks << " : Data Cache HIT!!!!!!!!!!! " << endl;
//#endif  
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    // Data Cache miss
    else 
    { 
        // Try to retrieve the block from the victim cache
        // addr >> block_offset is tag of victim cache
        block_t victim_block = victim_cache->remove(addr >> block_offset);
        
        // Victim Cache Hit
        if (victim_block.valid) 
        {
#ifdef DEBUG
	cout << *ticks << " : Found victim! " << endl;
#endif       
            if (block->valid) // If, Data cache block is valid, evict to Victim cache
            {
            	// Update the tag of the evicted block before inserting it to victim cache
            	uint64_t block_tag = (block->tag);
                block->tag = (block_tag << (set_offset-block_offset)|set_index);
                // insert block to victim cache
                victim_cache->insert(*block);
#ifdef DEBUG
	cout << *ticks << " : cache block eviction to victim : set = " << set_index
	     << " (tag = " << block->tag << ")" << endl;
#endif
            }
            // Restore tag before putting in to data cache
            victim_block.tag = (addr >> set_offset);
            // Place the victim block into the data cache
            blocks[set_index][0] = victim_block;
            // replay read
            read(m_inst);
        }
	// No hit in the victim cache
        else 
        { 
            // missed instruction = m_inst
            missed_inst = m_inst;
            // go to memory and find the proper block
            memory->load_block(addr & ~block_mask, block_size);
            // increment misses
            num_misses++;
#ifdef DEBUG
    cout << *ticks << " : cache miss : addr = " << addr
         << " (tag = " << tag << ", set = " << set_index << ")" << endl;
#endif
        }
    ///////////////////////////////////////////////////////////////////////////////////////////
    }
}

/************************************************************************
* write Operation (sd instruction)
*  Case1: Data Cache Hit (when, valid & tag is matched)
*	1) update block access time
*	2) block becomes dirty block
*	3) write rs2 value
*	4) number of store & accesses increase by 1
*  Case2: Data Cache Miss, Victim Cache Hit
*  	1) search victim cache components
*	2) when matched, 
*	   - update the Evicted block TAG and insert to Victim Cache
*	   - restore the TAG of victim block and insert to Data Cache
*	   - replay write
*	3) when miss-matched, go to Case3
*  Case3: Data Cache Miss, Victim Cache Miss
*	1) missed instruction = m_inst
*	2) go to memory and find the proper block
*	3) number of misses increase by 1
**************************************************************************/  

// Write data in memory.
void data_cache_t::write(inst_t *m_inst) {
    // Check the memory address alignment.
    uint64_t addr = m_inst->memory_addr;
    if(addr & 0b111) {
        cerr << "Error: invalid alignment of memory address " << addr << endl;
        exit(1);
    }

    // Calculate the set index and tag.
    uint64_t set_index = (addr & set_mask) >> block_offset;
    uint64_t tag = addr >> set_offset;
    
    // Check the direct-mapped cache entry.
    block_t *block = &blocks[set_index][0];

    // Cache hit
    if(block->valid && block->tag == tag) 
    { 
        // Update the last access time and dirty flag.
        block->last_access = *ticks;
        block->dirty = true;
        // Write a doubleword in the block.
        *(block->data + ((addr & block_mask) >> 3)) = m_inst->rs2_val;
        num_accesses++;
        num_stores++;
    }
    // Data Cache miss
    else 
    { 
        ///////////////////////////////////////////////////////////////////////////////////////////
        // Try to retrieve the block from the victim cache
        // addr >> block_offset is tag of victim cache
        block_t victim_block = victim_cache->remove(addr >> block_offset);
        
        if (victim_block.valid) // Victim Cache Hit
        {
            if (block->valid) // If, Data cache block is valid, evict to Victim cache
            {
            	uint64_t block_tag = (block->tag);
                // Update the tag of the evicted block before inserting it to victim cache
                block->tag = (block_tag << (set_offset-block_offset)|set_index);
                // insert block to victim cache
                victim_cache->insert(*block);
#ifdef DEBUG
	cout << *ticks << " : cache block eviction to victim : set = " << set_index
	     << " (tag = " << block->tag << ")" << endl;
#endif
            }
            // Restore tag before putting in to data cache
            victim_block.tag = (addr >> set_offset);
            // Place the victim block into the data cache
            blocks[set_index][0] = victim_block;
            // Replay write
            write(m_inst);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////
        else // No hit in the victim cache
        {
            // missed instruction = m_inst
            missed_inst = m_inst;
            // go to memory and find the proper block
            memory->load_block(addr & ~block_mask, block_size);
            // increment misses
            num_misses++;
#ifdef DEBUG
	cout << *ticks << " : cache miss : addr = " << addr
    	<< " (tag = " << tag << ", set = " << set_index << ")" << endl;
#endif
        }
    }
}

/************************************************************************
* Handle memory response
*  - Insert a block found in memory to Data cache
*  - If there is existing block in Data cache line
*  - Evict existing block to victim cache before inserting a block
*  
*  - After inserting a block found in memory to Data cache,
*  - Replay read or write operation depending on Instruction
**************************************************************************/  

// Handle a memory response.
void data_cache_t::handle_response(int64_t *m_data) {
    // Calculate the set index and tag.
    uint64_t addr = missed_inst->memory_addr;
    uint64_t set_index = (addr & set_mask) >> block_offset;
    uint64_t tag = addr >> set_offset;

    // Block replacement (Data cache block -> Victim cache)
    block_t *victim = &blocks[set_index][0];
    // when line is occupied by valid data cache block, evict block to victim cache
    if(victim->valid) {
#ifdef DEBUG
        cout << *ticks << " : Handle response : cache block eviction to victim : addr = " << addr
             << " (tag = " << tag << ", set = " << set_index << ")" << endl;
#endif
        // Update the tag of the evicted block before inserting it to victim cache
        uint64_t victim_tag = victim->tag;
        victim->tag = (victim_tag << (set_offset-block_offset)|set_index);
        // insert block to victim cache
        victim_cache->insert(*victim);
    }
    // Place the missed block.
    *victim = block_t(tag, m_data, /* valid */ true);

    // Replay the cache access.
    if(missed_inst->op == op_ld) { read(missed_inst); }
    else { write(missed_inst); }
    // Clear the missed instruction so that the cache becomes free.
    missed_inst = 0;
}

// Run data cache.
bool data_cache_t::run() {
    memory->run();          // Run the data memory.
    return missed_inst;     // Return true if the cache is busy.
}

// Print cache stats.
void data_cache_t::print_stats() {
    cout << endl << "Data cache stats:" << endl;
    cout.precision(3);
    cout << "    Number of loads = " << num_loads << endl;
    cout << "    Number of stores = " << num_stores << endl;
    cout << "    Miss rate = " << fixed
         << (num_accesses ? double(num_misses) / double(num_accesses) : 0)
         << " (" << num_misses << "/" << num_accesses << ")" << endl;
    cout.precision(-1);

    // Print victim cache stats.
    victim_cache->print_stats();
}

// Victim cache
victim_cache_t::victim_cache_t(uint64_t m_size) :
    num_entries(0),
    size(m_size),
    num_accesses(0),
    num_hits(0),
    num_writebacks(0),
    blocks(0) {
    blocks = new block_t[size]();
}

victim_cache_t::~victim_cache_t() {
    delete [] blocks;
}

/************************************************************************
* remove operation
*  - Assume, m_addr = tag of victim block
*    (when I called the function, I gave m_addr as a Tag)
*  - Find the block which Tag == m_addr using for loop
*  - accesses to victim cache occurred, increase num_accesses by 1
*  Case1: m_addr == Tag
*     1) block = matching block
*     2) shift all the blocks behind matching block
*     3) victim hit is considered as hit, increment number of hit by 1
*     4) as block is removed, num_entries decrease by 1
*     5) return block
*  Case2: m_addr != Tag
*     1) No matching block
*     2) return Invalid block
**************************************************************************/  

block_t victim_cache_t::remove(uint64_t m_addr) {
    block_t block;  // Invalid block, dummy block

    /******************************************************************************* 
       Search blocks[] if there's a matching entry. The matching entry should
       become the returning block, such as:

       block = blocks[i];

       Otherwise, the invalid block is returned to the data cache's query to
       indicate that there's no matching entry in the victim cache. 
    *********************************************************************************/

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    num_accesses++; // accesses to victim cache occurred, increase num_accesses by 1

    for (uint64_t i = 0; i < num_entries; ++i) {
        if ((blocks[i].tag == m_addr)) 
        {
            block = blocks[i];  // find the matching block

            // shift all the blocks behind matching block
            for (uint64_t j = i; j < num_entries - 1; ++j) 
            {
                blocks[j] = blocks[j + 1];
            }
            // victim hit is considered as hit
            num_hits++;
            // block is removed from victim 
            num_entries--;  

            return block;  // return matching block
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef DEBUG
	uint64_t tag = m_addr;
	uint64_t set = (m_addr & 0b111100000);
	cout << " remove : Victim cache miss : NO BLOCK FOUND!!! " << " / addr = " << m_addr
	<< " (tag = " << tag << " set index = " << set << ")" <<  endl;
#endif
    return block; // If we can't find the matching block, return Invalid block
    
}

/*********************************************************************************************************
* insert operation
*  - If the victim cache is full, the oldest entry is evicted and written back to the memory if dirty
*  - Victim cache full is when "num_entries == size"
*  - When oldest entry is evicted: num entries decrease by 1 & num writebacks increase by 1
*  - [Moreover, kite does not model writeback operation but simply traces the number of writebacks]
*
*  - The inserted block is placed at the end of the queue
*  - When Victim cache is full, shift remaing blocks by 1 and place the inserted block at the bottom
*  - When Victim cache in not full, place the inserted block at the bottom
*  - When block is inserted: num entries increase by 1
***********************************************************************************************************/  

void victim_cache_t::insert(block_t m_block) {
    /* If the victim cache is full, the oldest entry is evicted and written
       back to the memory if dirty, such as: */
    if(num_entries == size) 
    {
        if(blocks[0].dirty) { num_writebacks++; }
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        /************************************************************************
        The remaining entries are shifted forward, and the new victim block is
        placed at the end of FIFO queue, such as:
        *************************************************************************/

        // Shift remaining blocks
        for (uint64_t i = 0; i < num_entries - 1; ++i) {
            blocks[i] = blocks[i + 1];
        }
        num_entries--; // After shifting num_entries will decrease by one

        //////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    blocks[num_entries++] = m_block; // New block which is evicted from data cache will come in

#ifdef DEBUG
	uint64_t tag = m_block.tag;
	
	cout << " insert : Victim cache INSERTED! " << " Entry # " << num_entries
	<< " (tag = " << tag << ")" <<  endl;
#endif

}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void victim_cache_t::print_stats() {
    /* Do not modify this function */
    cout << endl << "Victim cache stats:" << endl;
    cout.precision(3);
    cout << "    Number of writebacks = " << num_writebacks << endl;
    cout << "    Hit rate = " << fixed
         << (num_accesses ? double(num_hits) / double(num_accesses) : 0)
         << " (" << num_hits << "/" << num_accesses << ")" << endl;
    cout.precision(-1);
}


