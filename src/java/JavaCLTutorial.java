/**
 * Created by Scott on 5/20/2014.
 */
import com.nativelibs4java.opencl.*;
import com.nativelibs4java.util.*;
import org.bridj.Pointer;

import java.nio.ByteOrder;

import static org.bridj.Pointer.*;
class JavaCLTutorial {
    public static void main(String[] args) throws Exception{
        long time = 0;

            long start = System.nanoTime();
            CLContext context = JavaCL.createBestContext();
            CLQueue queue = context.createDefaultQueue();
            ByteOrder byteOrder = context.getByteOrder();

            int n = 1024;
            Pointer<Float>
                    aPtr = allocateFloats(n).order(byteOrder),
                    bPtr = allocateFloats(n).order(byteOrder),
                    oPtr = allocateFloats(n).order(byteOrder);

            CLBuffer<Float>
                    a = context.createBuffer(CLMem.Usage.InputOutput, aPtr),
                    b = context.createBuffer(CLMem.Usage.InputOutput, bPtr),
                    out = context.createBuffer(CLMem.Usage.Output, oPtr);

            String src = IOUtils.readText(JavaCLTutorial.class.getResource("AddArrayzKernel.cl"));
            CLProgram program = context.createProgram(src);

            CLKernel addFloatsKernel = program.createKernel("add_floats");

            addFloatsKernel.setArgs(a, b, out, n);
            CLEvent addEvt = addFloatsKernel.enqueueNDRange(queue, new int[]{n});

            Pointer<Float> output = out.read(queue, addEvt);

            long finish = System.nanoTime();
            time += finish-start;
            System.out.println("time = " +time);


    }
}
