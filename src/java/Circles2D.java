import com.nativelibs4java.opencl.*;
import com.nativelibs4java.util.IOUtils;
import org.bridj.Pointer;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;

/**
 * Created by Scott on 10/21/2014.
 */
public class Circles2D {
    private CLContext context;
    private CLQueue queue;
    private ByteOrder byteOrder;
    private int circles;
    private ArrayList<Float>
            position,
            radius,
            velocity,
            mass;
    private Pointer<Float>
            circlePointer,
            velocityPointer,
            gravityPointer;
    private Pointer<Float> collisionPointer;
    private Pointer<Integer> collisionPairPointer;
    private Pointer<Integer> minAndMaxIndexPointer;
    private CLBuffer<Float>
            circleBuffer,
            velcityBuffer,
            gravityBuffer;
    private CLBuffer<Float> collisionBuffer;
    private CLBuffer<Integer> collisionPairBuffer;
    private CLBuffer<Integer> minAndMaxIndexBuffer;
    private String src;
    private CLProgram program;
    private CLKernel updatePositions;
    private CLKernel detectCollisions;
    private CLKernel resolveCollisions;
    private int potential_collisions;

    public Circles2D(boolean gpu){
        //boiler plate JavaCL code. Creates the context (chooses based on system)
        //then creates the queue for that context, to allow for kernel queueing
        if(gpu){
            context = JavaCL.createBestContext();
        }
        else{
            context = JavaCL.createBestContext(CLPlatform.DeviceFeature.CPU);
        }
        queue = context.createDefaultQueue();
        byteOrder = context.getByteOrder();
        position = new ArrayList<Float>();
        radius = new ArrayList<Float>();
        velocity = new ArrayList<Float>();
        mass = new ArrayList<Float>();
        circles = 0;
    }
    public void addCircle(float x, float y, float r, float mass) throws IllegalArgumentException{
        if(mass <= 0){
            throw new IllegalArgumentException("Mass cannot be a zero or non-positive integer");
        }
        this.position.add(x);
        this.position.add(y);
        this.radius.add(r);
        this.mass.add(mass);
        circles++;
    }
    public void init(float timeStep, int strategy, float restitution, float[] gravity) throws IOException{
        circlePointer = Pointer.allocateFloats(circles*4).order(byteOrder);
        velocityPointer = Pointer.allocateFloats(circles*2).order(byteOrder);
        gravityPointer = Pointer.allocateFloats(2).order(byteOrder);
        gravityPointer.set(0, gravity[0]);
        gravityPointer.set(1, gravity[1]);
        for(int i=0; i<circles; i++){
            circlePointer.set(4*i, position.get(2*i));
            circlePointer.set(4*i+1, position.get(2*i+1));
            circlePointer.set(4*i+2, radius.get(i));
            circlePointer.set(4*i+3, mass.get(i));

            velocityPointer.set(2*i, 0f/*(float)(Math.random()*0.4 - 0.2)*/);
            velocityPointer.set(2*i+1, 0f/*(float)(Math.random()*0.4 - 0.2)*/);
        }
        this.potential_collisions = circles*(circles-1)/2;
        collisionPointer = Pointer.allocateFloats(this.potential_collisions * 2).order(byteOrder);
        for (int i = 0; i < this.potential_collisions; i++) {
            collisionPointer.set(2 * i, 0f);
            collisionPointer.set(2 * i + 1, 0f);
        }
        circleBuffer = context.createBuffer(CLMem.Usage.InputOutput, circlePointer);
        velcityBuffer = context.createBuffer(CLMem.Usage.InputOutput, velocityPointer);
        gravityBuffer = context.createBuffer(CLMem.Usage.Input, gravityPointer);
        collisionBuffer = context.createBuffer(CLMem.Usage.InputOutput, collisionPointer);

        src = IOUtils.readText(new java.io.File("src/opencl/CirclePhysV2.cl"));
        program = context.createProgram(src);
        program.addBuildOption("-cl-finite-math-only");

        updatePositions = program.createKernel("update_position", circleBuffer, velcityBuffer, gravityBuffer,
                                               circles, timeStep, restitution);
        if(strategy == 0){
            detectCollisions = program.createKernel("detect_collisions", circleBuffer, collisionBuffer,
                                                    potential_collisions, circles);                                                             
            resolveCollisions = program.createKernel("resolve_collisions", circleBuffer, collisionBuffer,
                                                    velcityBuffer, potential_collisions, circles, restitution);
        }
        else if(strategy == 1){
		    //create collision pair buffer for strategy 1
		    collisionPairPointer = Pointer.allocateInts(this.potential_collisions * 2).order(byteOrder);
            int first = 0;
            for(int i = 0; i < circles-1; i++){
                int m = first;
                first += circles-1-i;
                for(int j=i+1; j<circles; j++){
                    collisionPairPointer.set((m+j-1-i)*2, i);
                    collisionPairPointer.set((m+j-1-i)*2 + 1, j);
                }
            }
            collisionPairBuffer = context.createBuffer(CLMem.Usage.Input, collisionPairPointer);
            detectCollisions = program.createKernel("detect_collisions_precomputed_pairs", circleBuffer, 
                                                    collisionBuffer, collisionPairBuffer, potential_collisions);     
            resolveCollisions = program.createKernel("resolve_collisions_with_pairs", circleBuffer, collisionBuffer,
                                                     velcityBuffer, collisionPairBuffer, potential_collisions,
                                                     restitution);
        }
        else{
            //create min and max collision per row buffer for strategy 2
		    minAndMaxIndexPointer = Pointer.allocateInts((circles) * 2).order(byteOrder);
            int min = 0;
            int max = circles-2;
            for(int i=0; i<=circles-1; i++){
                minAndMaxIndexPointer.set(i*2, min); 
                minAndMaxIndexPointer.set((i*2)+1, max);
                min = max+1;
                max += circles-2-i;
            }
            minAndMaxIndexBuffer = context.createBuffer(CLMem.Usage.Input, minAndMaxIndexPointer);
            detectCollisions = program.createKernel("detect_collisions_precomputed_rows", circleBuffer, 
                                                    collisionBuffer, minAndMaxIndexBuffer, potential_collisions,
                                                    circles);
            resolveCollisions = program.createKernel("resolve_collisions_with_rows", circleBuffer, collisionBuffer,
                                                     velcityBuffer, minAndMaxIndexBuffer, potential_collisions,
                                                     circles, restitution);
        }
    }
    public float[][] loop(){
        float[][] circs = new float[4][circles];
        CLEvent update = updatePositions.enqueueNDRange(queue, new int[]{circles});
        CLEvent detect = detectCollisions.enqueueNDRange(queue, new int[]{potential_collisions});
        CLEvent resolve = resolveCollisions.enqueueNDRange(queue, new int[]{potential_collisions});

        Pointer<Float> out = circleBuffer.read(queue, resolve);
        for(int i=0; i<circles; i++){
            circs[0][i] = out.get(4*i);
            circs[1][i] = out.get(4*i+1);
            circs[2][i] = out.get(4*i+2);
            circs[3][i] = out.get(4*i+3);
        }
        return circs;
    }
}
